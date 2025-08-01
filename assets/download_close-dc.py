import json
import wget

# URL = "https://nextcloud.mpi-klsb.mpg.de/index.php/s/HSWYzTreszKqf5Y/download?path=%2FCloSe-Di&downloadStartSecret=og97qdz97a&files="
# OUTPUT_DIR = "./close-di/"
# FILE_LIST = "close-di-unique-list.json"

URL = "https://nextcloud.mpi-klsb.mpg.de/index.php/s/HSWYzTreszKqf5Y/download?path=%2FCloSe-Dc&files="
OUTPUT_DIR = "./close-dc/"
FILE_LIST = "close-dc-unique-list.json"

from os import makedirs

makedirs(OUTPUT_DIR, exist_ok = True)

with open(FILE_LIST, 'r') as f:
    content = json.load(f)

print(len(content))

for file in content:
    url = URL + file + ".npz"
    print(url)
    # download the file
    wget.download(url, out=OUTPUT_DIR + file + ".npz")