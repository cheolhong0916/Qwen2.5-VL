import re

# Define placeholders for dataset paths

PRISM = {
    "annotation_path": "../../prefix_train_data_modified.json",
    "data_path": "../../",
}

SAT = {
    "annotation_path": "../../sat_train_data.json",
    "data_path": "../../",
}

# 2D
SPATIAL457_2D = {
    "annotation_path": "../../Spatial457_data/qwen_data/2D_tasks.json",
    "data_path": "../../",
}

# all
SPATIAL457_ALL = {
    "annotation_path": "../../Spatial457_data/qwen_data/Spatial457_all.json",
    "data_path": "../../",
}

PIXMO_POINTS = {
    "annotation_path": "../../pixmo_points_train_data.json",
    "data_path": "/",
}









# Fine-tuning-data
SYNTHETIC = {
    "annotation_path": "../../Fine-tuning-data/synthetic_SAT_Spatial457_PRISM_80.0k.json",
    "data_path": "../../",
}

REAL = {
    "annotation_path": "../../Fine-tuning-data/real_SPAR-7M_RoboSpatial_80.0k.json",
    "data_path": "../../",
}

STATIC = {
    "annotation_path": "../../Fine-tuning-data/static_RoboSpatial_PRISM_Spatial457_SAT_SPAR-7M_80.0k.json",
    "data_path": "../../",
}

DYNAMIC = {
    "annotation_path": "../../Fine-tuning-data/dynamic_Spatial457_SAT_SPAR-7M_80k.json",
    "data_path": "../../",
}

PERCEPTION = {  
    "annotation_path": "../../Fine-tuning-data/perception_Spatial457_SAT_SPAR-7M_80k.json",
    "data_path": "../../",
}

REASONING = {
    "annotation_path": "../../Fine-tuning-data/reasoning_RoboSpatial_PRISM_Spatial457_SAT_SPAR-7M_80.0k.json",
    "data_path": "../../",
}

_2D = {
    "annotation_path": "../../Fine-tuning-data/2d_Spatial457_SAT_SPAR-7M_80k.json",
    "data_path": "../../",
}

_3D = {
    "annotation_path": "../../Fine-tuning-data/3d_RoboSpatial_PRISM_Spatial457_SAT_SPAR-7M_80.0k.json",
    "data_path": "../../",
}


# w/o RoboSpatial
_3D_woRS = {
    "annotation_path": "../../Fine-tuning-data/woRS/3d_PRISM_Spatial457_SAT_SPAR-7M_80.0k.json",
    "data_path": "../../",
}

REAL_woRS = {
    "annotation_path": "../../Fine-tuning-data/woRS/real_SPAR-7M_80.0k.json",
    "data_path": "../../",
}

REASONING_woRS = {
    "annotation_path": "../../Fine-tuning-data/woRS/reasoning_PRISM_SPAR-7M_Spatial457_SAT_80.0k.json",
    "data_path": "../../",
}

STATIC_woRS = {
    "annotation_path": "../../Fine-tuning-data/woRS/static_PRISM_Spatial457_SAT_SPAR-7M_80.0k.json",
    "data_path": "../../",
}


# Fine-tuning-data + PIXMO
SYNTHETIC_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/synthetic_Fine-tuning-data_PIXMO_160k.json",
    "data_path": "../../",
}

REAL_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/real_Fine-tuning-data_PIXMO_160.0k.json",
    "data_path": "../../",
}

STATIC_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/static_RoboSpatial_PRISM_Spatial457_SAT_SPAR-7M_80.0k.json",
    "data_path": "../../",
}

DYNAMIC_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/dynamic_Fine-tuning-data_PIXMO_160k.json",
    "data_path": "../../",
}

PERCEPTION_PIXMO = {  
    "annotation_path": "../../Fine-tuning-data/perception_Fine-tuning-data_PIXMO_160k.json",
    "data_path": "../../",
}

REASONING_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/reasoning_Fine-tuning-data_PIXMO_160.0k.json",
    "data_path": "../../",
}

_2D_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/2d_Fine-tuning-data_PIXMO_160k.json",
    "data_path": "../../",
}

_3D_PIXMO = {
    "annotation_path": "../../Fine-tuning-data/3d_Fine-tuning-data_PIXMO_160.0k.json",
    "data_path": "../../",
}



# SAT30K
SAT30K = { 
    "annotation_path": "../../Fine-tuning-data/sat30k_SAT_30.0k.json",
    "data_path": "../../",
}

SAT10K_SPATIAL10K = {
    "annotation_path": "../../Fine-tuning-data/mix2_SAT_Spatial457_20.0k.json",
    "data_path": "../../",
}


CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "prism": PRISM,
    "sat": SAT,
    "pixmo_points": PIXMO_POINTS,
    "spatial457_2d": SPATIAL457_2D,
    "spatial457_all": SPATIAL457_ALL,
    "synthetic": SYNTHETIC,
    "real": REAL,
    "static": STATIC,
    "dynamic": DYNAMIC,
    "perception": PERCEPTION,
    "reasoning": REASONING,
    "2d": _2D,
    "3d": _3D,

    "3d_woRS": _3D_woRS,
    "real_woRS": REAL_woRS,
    "reasoning_woRS": REASONING_woRS,
    "static_woRS": STATIC_woRS,

    "synthetic_pixmo": SYNTHETIC_PIXMO,
    "real_pixmo": REAL_PIXMO,
    "static_pixmo": STATIC_PIXMO,
    "dynamic_pixmo": DYNAMIC_PIXMO,
    "perception_pixmo": PERCEPTION_PIXMO,
    "reasoning_pixmo": REASONING_PIXMO,
    "2d_pixmo": _2D_PIXMO,
    "3d_pixmo": _3D_PIXMO,

    "sat30k": SAT30K,
    "sat10k_spatial10k": SAT10K_SPATIAL10K,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    pass
    # dataset_names = ["cambrian_737k"]
    # dataset_names = ["pixmo_points"]
    # dataset_names = ['sat']
    # dataset_names = ['synthetic', 'dynamic', 'perception']
    # dataset_names = ['synthetic']
    # configs = data_list(dataset_names)
    # for config in configs:
    #     print(config)
