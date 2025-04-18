import tensorflow as tf
import numpy as np
from PIL import Image

# load model
model = tf.keras.models.load_model("flower_classifier.keras")

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


class_names = [
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

def predict_image(image_path):
    # 参数
    img_height = 180
    img_width = 180

    # 加载并预处理图像
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # 变成 batch size 为 1

    # 预测
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  # 如果你用的是 from_logits=True，就要 softmax 一下

    # 输出结果
    print("预测结果：这张图最可能是 '{}'，置信度为 {:.2f}%".format(
        class_names[np.argmax(score)-1], 100 * np.max(score)))

# 用你自己的图片调用
predict_image("test/daffodil.png")
