import os 
import shutil

old_lib = '/workspace/image_retrieval_dataset'
new_lib = '/workspace/image_retrieval_dataset_new'

os.makedirs(new_lib, exist_ok=True)

for folder in os.listdir(old_lib):
    folder_path = os.path.join(old_lib, folder)
    for file in os.listdir(folder_path):
        # copy and rename the file
        shutil.copy(os.path.join(folder_path, file), os.path.join(new_lib, f'{folder}_{file}'))
        
