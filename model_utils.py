from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

def load_model(name='resnet'):
    if name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess = preprocess_resnet
    else:
        base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
        preprocess = preprocess_xception
    return base_model, preprocess

def preprocess_image(image_path, target_size=(224, 224), preprocess_fn=None):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    if preprocess_fn:
        image = preprocess_fn(image)
    return image
