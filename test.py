import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# parameter
img_size = (180, 180)
img_height = 180
img_width = 180
temp_ds = tf.keras.utils.image_dataset_from_directory(
    "flower_photos17",
    image_size=(180, 180),
    batch_size=1
)
class_names = temp_ds.class_names
print(class_names)

# load model
model = tf.keras.models.load_model("flower_classifier_finetuned.keras")
sunflower_path = "test/img.png"


# predict function
def predict_image(image_path):
    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    top_k = tf.argsort(score, direction='DESCENDING')[:5]
    for i in top_k:
        print(f"{class_names[i]}: {score[i].numpy() * 100:.2f}%")


# run model
predict_image("test/img.png")