from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import hashlib
from PIL import Image
from typing import Union, Dict, List, Tuple, Any
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
# from ..download_pixmo_points_parse import apply_keyword_prompt, format_points, parse_point

import torch

seed = 42
rng = np.random.RandomState(seed)

POINT_PROMPT = [
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to the {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
]
def format_points(example):
        if "points" not in example:
            return None
        points = example["points"]
        # style = example["style"]
        if "label" in example:
            label = example["label"].lower()
        else:
            label = example["question"]
        if len(points) == 0:
            # if style in ["pointing", "point_count"]:
            return "There are none."
            # else:
            #     raise NotImplementedError()
        if "point_scale" in example:
            # Points are already normalized
            point_txt = points_to_text(points, example["point_scale"], label, label)
        else:
            # Points are in pixel coordinate
            h, w = example["image"].shape[:2]
            point_txt = points_to_text(points, [w/100, h/100], label, label)

        # if style == "point_count":
        #     return f"Counting the {point_txt} shows a total of {len(points)}."
        # else:
        #     return point_txt
        return point_txt
    
def apply_keywords(prompt, example, keywords):
    for keyword in keywords:
        res = prompt.split("{"+keyword+"}", maxsplit=2)
        prompt = res[0] + example[keyword] + res[1]
    return prompt

def apply_keyword_prompt(prompts, example, rng, keywords=None, dbg=False):
    if isinstance(prompts, list):
        assert keywords is None
        all_keywords = [sorted(re.findall("{([^{}]+)}", x)) for x in prompts]
        keywords = all_keywords[0]
        assert len(keywords) == len(set(keywords)), f"Repeated keywords in {keywords}"
        assert all(keywords == x for x in all_keywords), f"Inconsistent keywords in prompts {all_keywords}"
        assert not any("{" not in word[1:-1] and "}" in word[1:-1] for word in keywords)

        for k in keywords:
            assert k in example, f"Example missing expected field {k}, example={example}"

    if dbg:
        prompt = prompts[0]
    else:
        prompt = prompts[rng.randint(0, len(prompts))]
    return apply_keywords(prompt, example, keywords)



def compute_hash(string: Union[str, bytes]) -> str:
    if isinstance(string, str):
        return hashlib.sha256(string.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(string).hexdigest()
    
def points_to_text(points, scale, label_text, alt_text):
        if isinstance(scale, (tuple, list)):
            points /= np.array(scale)[None, :]
        else:
            points *= (100/scale)
        points = [[round(x, 5), round(y, 5)] for x, y in points]
        points.sort(key=lambda x: x[0]*10000 + x[1])
        if len(points) == 1:
            x_str, y_str = points[0]
            return f"<point x=\"{x_str:0.5f}\" y=\"{y_str:0.5f}\" alt=\"{alt_text}\">{label_text}</point>"
        point_text = []
        for ix, (x, y) in enumerate(points, start=1):
            point_text.append(f"x{ix}=\"{x:0.5f}\"")
            point_text.append(f"y{ix}=\"{y:0.5f}\"")
        point_text = " ".join(point_text)
        return f"<points {point_text} alt=\"{alt_text}\">{label_text}</points>"
    

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

import sys

assert len(sys.argv) > 1

if sys.argv[1] == "qwen":
    # ckpt_path = "/data/shared/Qwen/Qwen2.5-VL/qwen-vl-finetune/output/image_prefix_pixmo_points"
    ckpt_path = "Qwen/Qwen2.5-VL-7B-Instruct" # pretrained model
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto")
    model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(ckpt_path)
elif sys.argv[1] == "molmo":
    ckpt_path = "allenai/Molmo-7B-D-0924"
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)


def extract_points(text, image_w, image_h):
    all_points = []
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points

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

def is_point_in_region(point: Tuple[float, float], mask: np.ndarray) -> bool:
    """
    Check if the point (x, y) is within the region defined by the boolean mask.

    Parameters:
    - point (tuple of floats): x/y-coordinate of the point
    - mask (2D numpy array): Boolean mask of shape [H, W] representing the region

    Returns:
    - bool: True if the point is within the region, False otherwise
    """
    height, width = mask.shape
    x, y = point

    # Round the coordinates to the nearest integer
    x_int = int(round(x))
    y_int = int(round(y))

    # Check if the rounded point is within the bounds of the image
    if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
        return False

    # Check if the point is within the region
    return mask[y_int, x_int]

def is_valid_format(input_string):
    # Define the regular expression pattern
    pattern = re.compile(
        r'^(\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)\n?)+$|'
        r'^<point\s+x="\s*\d+(\.\d+)?"\s+y="\s*\d+(\.\d+)?"\s+alt="[\s\S]*?">[\s\S]*?</point>$|'
        r'^<points\s+(x\d+="\s*\d+(\.\d+)?"\s+y\d+="\s*\d+(\.\d+)?"\s+)+alt="[\s\S]*?">[\s\S]*?</points>$|'
        r'^<point\s+p=\s*\d{3}\s*,\s*\d{3}\s+alt="[\s\S]*?">[\s\S]*?</point>$|'
        r'^<points\s+(\d+=\s*\d{3}\s*,\s*\d{3}\s+)+alt="[\s\S]*?">[\s\S]*?</points>$'
    )

    # Match the entire input string against the pattern
    match = pattern.fullmatch(input_string.strip())

    # Return True if the match is successful, False otherwise
    return match is not None

def compute_precision(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        print(f"pred: {preds[i]}, mask shape: {masks[j].shape}")
        
        if is_point_in_region(preds[i], masks[j]):
            
            cnt += 1
    return cnt / len(preds)


def compute_recall(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(masks)


def f1_score(precision: float, recall: float, epsilon: float = 1e-10):
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall + epsilon)




split = "test"
pixmo_points_test_set = datasets.load_dataset("allenai/pixmo-points-eval", split=split)

total_correct_count = 0
total_precision = []
total_recall = []
total_f1 = []

for i, data in enumerate(pixmo_points_test_set):
    # print(data.keys())
    data["label"] = data["label"].lower()
    data["question"] = data["label"]
    
    data["point_scale"] = 100.
    points = data["points"]
    answer_points = data["points"] = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)

    
    hash_val = compute_hash(data["image_url"])
    # ith_image_save_dir = os.path.join(image_save_dir, f"{hash_val}.png")
    # ith_image_save_dir = os.path.join("/data/molmo/torch_datasets/pixmo_images", f"{hash_val}")
    ith_image_save_dir = os.path.join("/data/shared/Qwen/PIXMO_data/pixmo_images_testset", f"{hash_val}.png")

    if not os.path.exists(ith_image_save_dir):
        print(f"Failed to download image for example {i}")
        continue

    try:
        data["pil_image"] = Image.open(ith_image_save_dir)
        data["image"] = np.array(data["pil_image"])
        
    except Exception as e:
        print(f"Failed to open image for example {i}")
        continue

    # here = os.path.dirname(os.path.abspath(__file__))
    # img.save(os.path.join(here, "pixmo_image.png"))
    # print(example.keys())
    # print(example)

    prompt = apply_keyword_prompt(POINT_PROMPT, dict(data, label=data['label']), rng)
    system_content = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data['pil_image'],
                },
                {"type": "text", "text": prompt},
                
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        # images=data["pil_image"],
        images=image_inputs,
        videos=video_inputs,
        # padding=True,
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
    
    # print(f"output_text: {output_text[0]}")
    # break
    
    # print(f"prompt: {prompt}")
    
    
    ### GROUND TRUTH ###
    gt_answer = format_points(data)
    
    ### PREDICTION ###
    answer = output_text[0]
    
    print(f"GT answer: {gt_answer}")
    print(f"Predicted answer: {answer}")
    
    
    # break
    # print(f"answer: {answer}")
    abs_preds = extract_points(answer, data['image'].shape[1], data['image'].shape[0])
    print(f"extracted points: {abs_preds}")
    masks = np.array(data["masks"])
    # print(f"image shape: {data['image'].shape[1], data['image'].shape[0]}")
    # print(f"mask shape: {masks.shape}")
    
    if masks.shape[1] != data['image'].shape[0] or masks.shape[2] != data['image'].shape[1]:
        print(f"Mask shape {masks.shape} does not match image shape {data['image'].shape}")
        continue
    
    if len(answer_points) == 0:
        precision = recall = f1 = float(abs_preds is None or len(abs_preds) == 0)
        abs_gts = None
    else:
        abs_gts = answer_points
        abs_gts = abs_gts / 100.0 * np.array([data['image'].shape[1], data['image'].shape[0]])
        if not is_valid_format(answer):
            print("Invalid format for answer")
            precision = recall = f1 = 0.0
        else:
            abs_preds = np.array(abs_preds)
            dists = cdist(abs_preds, abs_gts)
            print(f"abs_preds: {abs_preds}, abs_gts: {abs_gts}, dists: {dists}")
            # print(masks.shape)
            # true_indices = [(i, j) for i, row in enumerate(masks[0]) for j, val in enumerate(row) if val]
            # print(f"true points: {true_indices}")
            row_ind, col_ind = linear_sum_assignment(dists)
            print(f"point candidates: {list(zip(row_ind, col_ind))}")
            precision = compute_precision(row_ind, col_ind, abs_preds, masks)
            recall = compute_recall(row_ind, col_ind, abs_preds, masks)
            f1 = f1_score(precision, recall)
            print("calculated")
    
    # print(f"image url: {data['image_url']}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    total_precision.append(precision)
    total_recall.append(recall)
    total_f1.append(f1)
    
    # break
    continue
    
    # predicted point
    # pred_point = parse_point(output_text[0])
    # if pred_point is None:
    #     print(f"Invalid prediction for sample {i}: {output_text[0]}")
    #     continue
    # pred_point /= 100  # convert to normalized coordinates
    # # print(f"Predicted point: {pred_point}")
    
    # # correct = check_grasp_prediction_accuracy(pred_point, 
    # #                                          data['grasp_point_px_candidates'], 
    # #                                          data['grasp_candidiates'], 
    # #                                          data['gt_grasp'])
    # # print(f"Prediction accuracy: {correct}")
    # total_correct_count += int(correct)
    # if i % 10 == 0:
    #     print(f"[Processed {i} samples] current accuracy: {total_correct_count / (i + 1):.2f}")
    
print(f"Final precision: {np.mean(total_precision):.4f} ({len(total_precision)})")
print(f"Final recall: {np.mean(total_recall):.4f} ({len(total_recall)})")
print(f"Final F1 score: {np.mean(total_f1):.4f} ({len(total_f1)})")
print(f"Final accuracy: {total_correct_count / len(prism_test_set):.2f} ({total_correct_count}/{len(prism_test_set)})")
