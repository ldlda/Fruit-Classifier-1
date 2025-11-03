import json
import typing
from typing import Any, Callable, TypedDict

import av
import cv2
import keras  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import streamlit as st
import tensorflow as tf
from keras.models import Model  # type: ignore[import]
from PIL import Image
from streamlit_webrtc import VideoProcessorBase


# --- MODEL CONFIGURATION ---
class ModelConfig(TypedDict):
    "model config shape"

    file: str
    size: tuple[int, int]
    family: str
    last_conv_layer: str


IMAGE_EXTENSIONS: list[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".tif",
    ".tiff",
    ".bmp",
    ".psd",
    ".ai",
    ".eps",
    ".raw",
    ".cr2",
    ".nef",
    ".dng",
]
"some common image extensions"

# Import the specific preprocessing functions

MODEL_CONFIG: dict[str, ModelConfig] = {
    "MobileNetV2": {
        "file": "mobilenet_model.keras",
        "size": (224, 224),
        "family": "mobilenet_v2",
        "last_conv_layer": "out_relu",
    },
    "EfficientNetV2B0": {
        "file": "efficientnet_model.keras",
        "size": (224, 224),
        "family": "efficientnet_v2",
        "last_conv_layer": "top_conv",
    },
}
"model config, with keys being the model names and values conforming to [ModelConfig] schema"

# --- CACHED HELPER FUNCTIONS ---


@st.cache_resource
def load_my_model(model_path_in: str) -> Model:
    loaded_model = keras.models.load_model(model_path_in)
    assert loaded_model is not None
    return typing.cast(Model, loaded_model)


@st.cache_data
def load_my_labels() -> dict[int, str]:
    with open("labels.json", "r", encoding="utf-8") as f:
        labels_from_json = json.load(f)
        label_map = {int(k): v for k, v in labels_from_json.items()}
    return label_map


# --- PREPROCESSING FUNCTIONS ---


def get_preprocess_fn(family: str) -> Callable[[Any], Any]:
    if family == "mobilenet_v2":
        return keras.applications.mobilenet_v2.preprocess_input

    if family == "efficientnet_v2":
        return keras.applications.efficientnet_v2.preprocess_input

    raise ValueError(f"Unsupported model family '{family}'.")


def preprocess_image(
    img_pil: Image.Image,
    size: tuple[int, int],
    preprocess_fn: Callable[[Any], Any],
) -> np.ndarray:
    # Ensure RGB
    img = img_pil.convert("RGB")

    # Resize the image
    img = img.resize(size)

    # Convert to numpy array
    img_array = np.array(img)

    # Add the "batch" dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply model-specific preprocessing
    img_array = preprocess_fn(img_array)

    return img_array


# --- GRAD-CAM LOGIC ---


def make_gradcam_heatmap(
    img_array: np.typing.NDArray,
    model: Model,
    last_conv_layer_name: str,
    pred_index: int | tf.Tensor | None = None,
) -> np.typing.NDArray:
    """Compute a Grad-CAM heatmap for a given input and model.

    Steps (high level):
    1) Build a sub-model that outputs both the last conv layer feature maps and the final predictions.
    2) Forward-pass the image to get those tensors.
    3) If no class index is provided, pick the top-predicted class.
    4) Compute gradients of the selected class score w.r.t. the conv feature maps.
    5) Global-average-pool the gradients across spatial dims to obtain channel weights.
    6) Weight the conv feature maps by those channel weights and sum over channels.
    7) ReLU and normalize to [0, 1].
    """
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])

        if isinstance(predictions, list):
            predictions_tensor = predictions[0]
        else:
            predictions_tensor = predictions

        if pred_index is None:
            pred_index = tf.argmax(predictions_tensor[0])

        class_channel = predictions_tensor[:, pred_index]

    # Gradients of the target class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Global-average pool gradients over spatial dims -> channel weights, shape (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Feature maps for this image, shape (H, W, C)
    conv_outputs = conv_outputs[0]

    # Channel-wise weighted sum: sum_c (A[:, :, c] * w[c]) -> shape (H, W)
    # This is equivalent to a tensordot over the channel axis but clearer for type checkers
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# --- GRAD-CAM DISPLAY FUNCTION ---


def generate_gradcam_overlay(
    img_pil: Image.Image, heatmap: np.ndarray, alpha=0.4
) -> npt.NDArray[np.uint8]:
    """
    Overlays a heatmap on the original PIL image.
    This works IN-MEMORY, no files are saved.
    """
    # Convert original PIL image to NumPy array (RGB)
    img_rgb = np.array(img_pil.convert("RGB"))

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    # Rescale heatmap to 0-255 and apply colormap
    heatmap_8bit = (255 * heatmap_resized).astype(np.uint8, copy=False)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

    # Convert heatmap to RGB.
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend the original image with the heatmap
    superimposed_img = cv2.addWeighted(heatmap_rgb, alpha, img_rgb, 1 - alpha, 0)

    return superimposed_img.astype(np.uint8, copy=False)


# --- VIDEO PROCESSING CLASS ---


class FruitClassifierProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        # Load the model and labels
        self.model_name = st.session_state.get("model_choice", "MobileNetV2")
        self.model_config = MODEL_CONFIG[self.model_name]
        self.model = load_my_model(self.model_config["file"])
        self.labels = load_my_labels()
        self.preprocess_fn = get_preprocess_fn(self.model_config["family"])
        self.size = self.model_config["size"]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame from the stream into a NumPy array
        img = frame.to_ndarray(format="bgr24")

        # --- PREPROCESS THE FRAME ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, self.size)

        # Batch dimension
        img_array = np.expand_dims(img_resized, axis=0)

        # Apply model-specific preprocessing
        img_preprocessed = self.preprocess_fn(img_array)

        # --- PREDICT ---
        predictions = self.model.predict(img_preprocessed)[0]
        pred_index = np.argmax(predictions)
        pred_label = self.labels[int(pred_index)]
        confidence = np.max(predictions)

        # --- DRAW ON FRAME ---
        # Draw the prediction text on the *original* BGR frame
        text = f"{pred_label} ({confidence:.2f})"

        cv2.putText(
            img,  # Draw on the original BGR frame
            text,
            (10, 30),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),  # Color (Green)
            2,  # Thickness
        )

        # Convert the modified BGR frame back to the stream format
        return av.VideoFrame.from_ndarray(img, format="bgr24")
