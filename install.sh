conda init
source ~/.bashrc

pip install -U fashion-clip
python find_closest_images.py

pip install gdown 
gdown 1YuI8OMiqQXPozVBi8UKczWlCqkI38dSU -d assets 
cd assets
unzip close-di.zip
rm close-di.zip