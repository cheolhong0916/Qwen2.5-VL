import os

import re
import xml.etree.ElementTree as ElementTree
import datasets
import huggingface_hub as hf_hub
import h5py
from PIL import Image
import numpy as np
import yaml
from typing import Optional

from qwen2_vl.model import Qwen2VLChat
from dataset_utils import dump_image


def parse_point(pred: str, image_size: Optional[tuple[int, int]] = None):
    """
    Args:
        pred: The prediction string from the model.
        image_size: The size of the image, (width, height). If provided, return in pixels, otherwise return in normalized coordinates.
    Returns:
        The predicted point as a numpy array of shape (2,).
    """
    point_xmls = re.findall(r'<points?.*?</points?>', pred, re.DOTALL)
    if len(point_xmls) == 0:
        print(f"Invalid prediction: {pred}")
        return None
    point_xml = point_xmls[0]
    try:
        point_elem = ElementTree.fromstring(point_xml)
        
        if point_elem is not None:
            if point_elem.tag == 'point':
                x = float(point_elem.get('x'))
                y = float(point_elem.get('y'))
            elif point_elem.tag == 'points':
                x = float(point_elem.get('x1'))
                y = float(point_elem.get('y1'))
            else:
                print(f"Invalid prediction: {pred}")
                return None
            ret = np.array([x, y])
            if image_size is not None:
                ret = ret / 100 * np.array(image_size)
            return ret
        else:
            print("No point element found in XML")
    except ElementTree.ParseError as e:
        print(f"Failed to parse XML: {e}")
    return None


def point_to_xml(grasp_pt: np.ndarray):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = "Where to grasp the object"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

# for evaluation
def map_sample(file_loc_map: dict[str, str], ex: dict):
    h5_path = file_loc_map[ex["scene_path"]]
    grasp_pt_px_lst = []
    with h5py.File(h5_path, "r") as f:
        img = Image.fromarray(f[ex["view_id"]]["rgb"][:])
        grasp_pt_px = f[ex["view_id"]][ex["obs_id"]]["grasp_point_px"][:]
        grasp_pt_px = grasp_pt_px / np.array([img.width, img.height])
        for obs_id in f[ex["view_id"]].keys():
            # print(obs_id)
            if obs_id.startswith("obs_"):
                grasp_pt_px_lst.append(f[ex["view_id"]][obs_id]["grasp_point_px"][:] / np.array([img.width, img.height]))
        annotation = yaml.safe_load(f[ex["view_id"]][ex["obs_id"]]["annot"][()])
        grasp_id = annotation["grasp_id"]
        # print(annotation["object_category"], annotation["object_id"], annotation["grasp_id"])
        # print(f[ex["view_id"]].keys(), ex["obs_id"])
        grasp_candidiates = [o for o in f[ex["view_id"]].keys() if o.startswith("obs_")]

    task = ex["task"]
    prompt = f"Point to the grasp that would accomplish the following task: {task}"
    point_xml = point_to_xml(grasp_pt_px)
    response = f"In order to accomplish the task \"{task}\", the optimal grasp is described as follows: \"{ex['matching_grasp_desc']}\".\n\n{point_xml}"
    
    gt_grasp = ex["obs_id"]

    return dict(
        image=img,
        prompt=prompt,
        text=response,
        style="pointing",
        grasp_candidiates=grasp_candidiates,
        gt_grasp=gt_grasp,
        grasp_point_px_candidates=grasp_pt_px_lst,
    )

def build_pointing_dataset(split: str, num_proc: int = 10) -> datasets.Dataset:
    hf_fs = hf_hub.HfFileSystem()
    chunks = hf_fs.ls(f"datasets/allenai/PRISM/PRISM-{split}", detail=False)
    urls = []
    for chunk in chunks:
        path = chunk[len("datasets/allenai/PRISM/"):]
        urls.append(hf_hub.hf_hub_url(repo_id="allenai/PRISM", filename=path, repo_type="dataset"))

    dl_manager = datasets.DownloadManager(dataset_name="allenai/PRISM", record_checksums=False)
    paths = dl_manager.download_and_extract(urls)

    file_loc_map = {}
    for path in paths:
        path = str(path)
        for file in os.listdir(path):
            file_loc_map[file] = os.path.join(path, file)

    metadata_ds = datasets.load_dataset("allenai/PRISM", split=split)
    dataset = metadata_ds.map(lambda ex: map_sample(file_loc_map, ex), num_proc=num_proc)
    return dataset


def check_grasp_prediction_accuracy(pred_grasp_pt_px, grasp_point_px_candidates, grasp_candidates, gt_grasp):
    # find argmin distance(pred_grasp_pt_px, grasp_point_px_candidates) index
    pred_grasp_pt_px = np.array(pred_grasp_pt_px)
    grasp_point_px_candidates = np.array(grasp_point_px_candidates)
    distances = np.linalg.norm(grasp_point_px_candidates - pred_grasp_pt_px, axis=1)
    pred_idx = np.argmin(distances)
    pred_grasp = grasp_candidates[pred_idx]
    print(f"Predicted grasp: {pred_grasp}, GT grasp: {gt_grasp}")
    return pred_grasp == gt_grasp


def dump_image_func(line):
    return dump_image(line, img_root)
    
model = Qwen2VLChat(
        model_path="/data/shared/Qwen/Qwen2.5-VL/qwen-vl-finetune/output",
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        use_custom_prompt=True,
        min_pixels=1280*28*28,
        max_pixels=5120*28*28
    )
model.set_dump_image(dump_image_func)

prism_test_set = build_pointing_dataset("test")

for data in prism_test_set:
    print(data["prompt"])
    print(f"GT point: {parse_point(data['text'])}")
    print(f"GT grasp: {data['gt_grasp']}")
    print(f"Grasp candidates: {data['grasp_candidiates']}")
    
    # model point predicition
    # [{'type': 'text', 'value': 'Question: Assume accounts have normal balances, 
    #                                       solve for the one missing account balance: Dividends. Equipment was recently purchased, so there is neither depreciation expense nor accumulated depreciation. 
    #                               '}, 
    # {'type': 'image', 'value': '/data/mmmu/images/MMMU/387.jpg'}, 
    # {'type': 'text', 'value': '\nOptions:\nA. $194,815\nB. $182,815\nC. $12,000\nD. $9,000\nPlease select the correct answer from the options above.'}]
    
    break

