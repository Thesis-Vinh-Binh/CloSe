from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import pandas as pd 
import os 


INPUT = '/workspace/CloSe/data/close_image_scan'
LIBRARY = '/workspace/image_retrieval_dataset'

fclip = FashionCLIP('fashion-clip')

library_images = []
for folder in os.listdir(LIBRARY):
    for image in os.listdir(os.path.join(LIBRARY, folder)):
        library_images.append(os.path.join(LIBRARY, folder, image))

input_images = []
for image in os.listdir(INPUT):
    if image.endswith('.png'):
        input_images.append(os.path.join(INPUT, image))

input_embeddings = fclip.encode_images(input_images, batch_size=16)
library_embeddings = fclip.encode_images(library_images, batch_size=64)

# find the 5 closest image in the library for each input image
output = pd.DataFrame(columns=['input_image', 'closest_images'])

for i in range(len(input_images)):
    arr = np.linalg.norm(input_embeddings[i] - library_embeddings, axis=1)
    # sort the library images filename by the distance
    sorted_library_images = [library_images[j] for j in np.argsort(arr)]
    output.loc[i] = [input_images[i], sorted_library_images[:5]]
output.to_csv('closest_images.csv', index=False)








