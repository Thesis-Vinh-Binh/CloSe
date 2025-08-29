from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import pandas as pd 
import os 
import argparse

def get_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, default="/workspace/CloSe/data/close_image_scan", help="path to the input folder")
    parser.add_argument("--library_folder", "-l", type=str, default="/workspace/image_retrieval_dataset_new", help="path to the library folder")
    return parser.parse_args()

args = get_command_args()

INPUT = args.input_folder
LIBRARY = args.library_folder

fclip = FashionCLIP('fashion-clip')

print('Loading library images')
library_images = []
for folder in os.listdir(LIBRARY):
    library_images.append(os.path.join(LIBRARY, folder))
print(f'Loaded {len(library_images)} library images')
library_captions = pd.read_csv(f'/workspace/ImageCaption/logs/caption/captions.csv')

print('Loading input images')
input_images = []
for image in os.listdir(INPUT):
    if image.endswith('.png'):
        input_images.append(os.path.join(INPUT, image))
print(f'Loaded {len(input_images)} input images')
input_captions = pd.read_csv(f'{INPUT}/captions.csv')        

input_embeddings = fclip.encode_images(input_images, batch_size=16)
text_input_embeddings = fclip.encode_text(input_captions['caption'], batch_size=16)
library_embeddings = fclip.encode_images(library_images, batch_size=64)
text_library_embeddings = fclip.encode_text(library_captions['caption'], batch_size=16)

# find the 5 closest image in the library for each input image
output = pd.DataFrame(columns=['input_image', 'closest_images'])

for i in range(len(input_images)):
    img_img = np.linalg.norm(input_embeddings[i] - library_embeddings, axis=1)
    img_text = np.linalg.norm(input_embeddings[i] - text_library_embeddings, axis=1)
    text_img = np.linalg.norm(text_input_embeddings[i] - library_embeddings, axis=1)
    text_text = np.linalg.norm(text_input_embeddings[i] - text_library_embeddings, axis=1)
    arr = img_img + img_text + text_img + text_text
    # sort the library images filename by the distance
    sorted_library_images = [library_images[j] for j in np.argsort(arr)]
    output.loc[i] = [input_images[i], sorted_library_images[:5]]
output.to_csv(f'{INPUT}/closest_images.csv', index=False)








