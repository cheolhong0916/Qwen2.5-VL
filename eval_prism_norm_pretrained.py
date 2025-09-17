from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/data/shared/Qwen/Qwen2.5-VL/qwen-vl-finetune/output/image_prefix_sat_prism", torch_dtype="auto", device_map="auto"
# )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)


# processor = AutoProcessor.from_pretrained("/data/shared/Qwen/Qwen2.5-VL/qwen-vl-finetune/output/image_prefix_sat_prism")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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
    # prompt = f"Detech the grasp point that would accomplish the following task: {task}. Return their locations in the form of coordinates. The format of output should be like <point x='x1' y='x2' alt='Where to grasp the object'>Where to grasp the object</point>"
    prompt = f"Point to the grasp that would accomplish the following task: {task} \n\nPlease answer using normalized coordinates (0–100) in this format <point x=\"...\" y=\"...\" alt='Where to grasp the object'>Where to grasp the object</point>"
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
    gt_grasp_idx = grasp_candidates.index(gt_grasp)
    gt_grasp_pt_px = grasp_point_px_candidates[gt_grasp_idx]
    # print(f"Predicted grasp point: {pred_grasp_pt_px}, GT grasp point: {gt_grasp_pt_px}")
    # print(f"Predicted grasp: {pred_grasp}, GT grasp: {gt_grasp}")
    return pred_grasp == gt_grasp


prism_test_set = build_pointing_dataset("test")

total_correct_count = 0

for i, data in enumerate(prism_test_set):
    # print(data["prompt"])
    # print(f"GT point: {parse_point(data['text'])}")
    # print(f"GT grasp: {data['gt_grasp']}")
    # print(f"Grasp candidates: {data['grasp_candidiates']}")
    # print(f"Grasp point candidates: {data['grasp_point_px_candidates']}")
    # print(f"PIL image: {data['image']}")
    
    # break/
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": data["prompt"] + "\n"},
    #             {
    #                 "type": "image",
    #                 "image": data['image'],
    #             },
    #         ],
    #     }
    # ]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data['image'],
                },
                {"type": "text", "text": "\n" + data["prompt"]},
                
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # print(output_text)
    gt_grasp_idx = data['grasp_candidiates'].index(data['gt_grasp'])
    gt_grasp_pt_px = data['grasp_point_px_candidates'][gt_grasp_idx]
    # print(f"GT grasp point: {gt_grasp_pt_px}")
    
    # predicted point
    pred_point = parse_point(output_text[0]) # image_size=(data['image'].width, data['image'].height))
    if pred_point is None:
        print(f"Invalid prediction for sample {i}: {output_text[0]}")
        continue
    pred_point /= 100  # convert to normalized coordinates
    # print(f"Predicted point: {pred_point}")
    
    correct = check_grasp_prediction_accuracy(pred_point, 
                                             data['grasp_point_px_candidates'], 
                                             data['grasp_candidiates'], 
                                             data['gt_grasp'])
    # print(f"Prediction accuracy: {correct}")
    total_correct_count += int(correct)
    if i % 10 == 0:
        print(f"[Processed {i} samples] current accuracy: {total_correct_count / (i + 1):.2f}")
    
print(f"Final accuracy: {total_correct_count / len(prism_test_set):.2f} ({total_correct_count}/{len(prism_test_set)})")
