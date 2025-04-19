import os
import shutil
import scipy.io
from pathlib import Path
import re
# set path
image_dir = Path('17flowers')
label_file = Path('datasplits.mat')
output_dir = Path('flowers_photos120')

# make dir
output_dir.mkdir(parents=True, exist_ok=True)


mat = scipy.io.loadmat(label_file)
labels = mat['labels'][0]


flower_names = [
    "daffodil", "snowdrop", "lily_valley", "bluebell", "crocus", "iris", "tigerlily", "tulip", "fritillary",
    "sunflower", "daisy", "colts_foot", "dandelion", "cowslip", "buttercup", "windflower", "pansy"
]


for index in range(len(flower_names)):
    name = flower_names[index]
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    name_split = name.split(' ')
    for inName in range(len(name_split)):
        name_split[inName] = name_split[inName].capitalize()
    flower_names[index]=name_split
    flower_names[index] = ''.join(name_split)
print(flower_names)

for name in flower_names:
    class_dir = output_dir / name
    class_dir.mkdir(parents=True, exist_ok=True)


for idx, label in enumerate(labels):
    image_filename = f'image_{idx + 1:05d}.jpg'
    src_path = image_dir / image_filename
    flower_name = flower_names[label - 1]  # fix bug here
    output_path = output_dir / flower_name / image_filename
    shutil.copy(src_path, output_path)
