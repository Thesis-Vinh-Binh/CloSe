import os
import pandas as pd
import argparse

def get_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", "-d", type=str, required=True, help="path to the dataset folder")
    return parser.parse_args()

args = get_command_args()
path = f'/workspace/CloSe/data/{args.data_folder}'

files = os.listdir(path)
images = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
image_names = [os.path.splitext(file)[0] for file in images]

captions = pd.read_csv(f'{path}/captions.csv')

# iterate over captions and check if the filename match with any in images, replace filename by the filename in images
for index, row in captions.iterrows():
    filename = row['filename']
    if filename in image_names:
        captions.at[index, 'filename'] = images[image_names.index(filename)]
    else:
        print(f'{filename} not found in {path}')
        
captions_new = captions['filename'].values
for image in images:
    if image not in captions_new:
        print(f'{image} not found in captions, hence remove')
        os.remove(f'{path}/{image}')

captions.to_csv(f'{path}/captions.csv', index=False)
print(f"Done and saved to {path}/captions.csv")

    