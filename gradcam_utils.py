import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    import tensorflow as tf
    import numpy as np

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # Convert conv_outputs and grads to numpy
    conv_outputs = conv_outputs[0].numpy()
    grads = grads[0].numpy()

    # Weight the channels by importance
    weights = np.mean(grads, axis=(0, 1))

    # Multiply each channel in feature map array by its corresponding weight
    for i in range(weights.shape[-1]):
        conv_outputs[:, :, i] *= weights[i]

    heatmap = np.mean(conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-10  # avoid divide by zero
    return heatmap


def overlay_heatmap(heatmap, original_img_path, alpha=0.5):
    image = cv2.imread(original_img_path)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, alpha, color, 1 - alpha, 0)
    return superimposed


