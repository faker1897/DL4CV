import os
import shutil
import scipy.io
from pathlib import Path
import re
# set path
image_dir = Path('jpg')
label_file = Path('imagelabels.mat')
output_dir = Path('flowers_photos')

# make dir
output_dir.mkdir(parents=True, exist_ok=True)


mat = scipy.io.loadmat(label_file)
labels = mat['labels'][0]


flower_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
    'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger',
    'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian',
    'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
    'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
    'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil',
    'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup',
    'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris',
    'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple',
    'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia',
    'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss',
    'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia',
    'blanket flower', 'trumpet creeper', 'blackberry lily'
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
