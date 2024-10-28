import os

os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobilevit_v2 import build_MobileViT_v2


# Load the labels (ImageNet class names)
os.system("wget -q -O imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


def test_prediction(*, image_path, model=None, image_shape=(224, 224), show=False):
    # Load and process the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = img[:, :, ::-1]

    img = cv2.resize(img, image_shape)

    if show:
        plt.imshow(img)

    img = img  / 255. # Normalize pixel values to [0, 1]
    img = img.astype("float32")  # Ensure the correct type for TensorFlow
    # Add the batch dimension
    img = np.expand_dims(img, 0)  # Shape becomes (1, 256, 256, 3)

    # Perform prediction
    preds = model.predict(img, verbose=0)

    # Output prediction
    print(f"Model: {model.name}, Predictions: {labels[preds.argmax()]}")


keras_model_0 = build_MobileViT_v2(
    width_multiplier=2.0,
    input_shape=(256, 256, 3),
    pretrained=True,
    pretrained_weight_name="keras_mobilevitv2-im1k-256-2.0.weights.h5",
)

keras_model_1 = build_MobileViT_v2(
    width_multiplier=1.0,
    input_shape=(256, 256, 3),
    pretrained=True,
    pretrained_weight_name="keras_mobilevitv2-im1k-256-1.0.weights.h5",
)

keras_model_2 = build_MobileViT_v2(
    width_multiplier=0.5,
    input_shape=(256, 256, 3),
    pretrained=True,
    pretrained_weight_name="keras_mobilevitv2-im1k-256-0.5.weights.h5",
)

# Download an example image
os.system("wget -q -O example.jpg https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg")

show = True
for keras_model, img_shape in (
    (keras_model_0, (256, 256)),
    (keras_model_1, (256, 256)),
    (keras_model_2, (256, 256)),
):
    test_prediction(image_path=r"example.jpg", model=keras_model, image_shape=img_shape, show=show)
    print("--------------")