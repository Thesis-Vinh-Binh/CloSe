# from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import pandas as pd
import os 
import json


DATA_PATH = "/workspace/CloSe/data"
EVAL_PATH = "/workspace/data/3d_data"
eval_dict = {
    "close_image_scan": {
        "cg_baseline": [],
        "cg_cot": [],
        "cg_blip": [],
        "cg_retrieval": []
    },
    "s_sketch": {
        "cg_baseline": [],
        "cg_cot": [],
        "cg_blip": [],
    },
    "example_data": {
        "cg_retrieval": [],
        "cg_cot": [],
        "cg_blip": []
    }
}

dataset_dict = {}
    


def all_sub_images(path):
    files = os.listdir(path)
    sub_images = []
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(path, file)
            sub_images.append(file_path)
    return sub_images


def get_all_sub_dirs(path):
    dir_list = os.listdir(path)
    sub_dirs = []
    for file in dir_list:
        sub_path = os.path.join(path, file)
        if os.path.isdir(sub_path):
            sub_dirs.append(sub_path)
    return sub_dirs

if __name__ == "__main__":
    DATASET_JSON = "/workspace/CloSe/data/dataset_dict.json"
    EVAL_JSON = "/workspace/CloSe/data/eval_dict.json"
    if os.path.exists(DATASET_JSON):
        with open(DATASET_JSON, "r") as f:
            dataset_dict = json.load(f)
    else:
        dataset_dict = {}

        for dataset in eval_dict.keys():
            dataset_dict[dataset] = all_sub_images(os.path.join(DATA_PATH, dataset))
            print(f"Loaded {len(dataset_dict[dataset])} {dataset} sub directories")

        with open(DATASET_JSON, "w") as f:
            json.dump(dataset_dict, f, indent=4)
            
    if os.path.exists(EVAL_JSON):
        with open(EVAL_JSON, "r") as f:
            eval_dict = json.load(f)
    else:
        for dataset, methods in eval_dict.items():
            for method, _ in methods.items():
                path = os.path.join(EVAL_PATH, f"{dataset}_{method}", "vis_new")
                sub_dirs = get_all_sub_dirs(path)
                for sub_dir in sub_dirs:
                    eval_dict[dataset][method].append(sub_dir)
        with open(EVAL_JSON, "w") as f:
            json.dump(eval_dict, f, indent=4)

#     # fclip = FashionCLIP('fashion-clip')

# import os 
# import shutil
# def get_all_sub_dirs(path):
#     dir_list = os.listdir(path)
#     sub_dirs = []
#     for file in dir_list:
#         sub_path = os.path.join(path, file)
#         if os.path.isdir(sub_path):
#             sub_dirs.append(sub_path)
#     return sub_dirs

# PATH = "/workspace/data/3d_data/example_data_cg_cot/vis_new"
# OUTPUT = "/workspace/CloSe/data/example_data/"
# sub_dirs = get_all_sub_dirs(PATH)

# os.makedirs(OUTPUT, exist_ok=True)
# for sub_dir in sub_dirs:
#     new_name = os.path.basename(sub_dir)[14:] + ".png"
#     path = os.path.join(sub_dir, "gt_image.png")
#     # copy the file to OUTPUT
#     shutil.copy(path, os.path.join(OUTPUT, new_name))