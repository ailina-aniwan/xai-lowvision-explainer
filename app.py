import io
from typing import Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from gtts import gTTS

from utils import (
    load_model_and_transform,
    GradCAM,
    overlay_heatmap_on_image,
    get_spatial_description,
)


st.set_page_config(
    page_title="Accessible Object Explainer for Low-Vision Users",
    layout="wide",
)


@st.cache_resource
def get_model_and_helpers():
    """
    Load the pretrained ResNet18 model, transform pipeline, and class names
    once and cache them for all Streamlit runs.
    """
    model, transform, class_names = load_model_and_transform()
    target_layer = model.layer4[-1].conv2  # last conv layer for Grad-CAM
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    return model, transform, class_names, grad_cam


def generate_audio_bytes(text: str, lang: str = "en") -> io.BytesIO:
    """
    Use gTTS to convert text to an in-memory MP3 audio stream.
    """
    audio_fp = io.BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp


def analyze_image(
    image: Image.Image,
    model,
    transform,
    class_names,
    grad_cam: GradCAM,
) -> Tuple[str, float, np.ndarray]:
    """
    Run the model + Grad-CAM on an input PIL image.

    Returns:
        label (str): Predicted ImageNet class name.
        prob (float): Confidence for that class.
        cam (np.ndarray): 2D Grad-CAM heatmap (values in [0, 1]).
    """
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

    # Move to CPU (model should already be on CPU)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # Use Grad-CAM helper to get heatmap and prediction
    cam, class_idx, prob = grad_cam(input_tensor)
    label = class_names[class_idx]

    return label, prob, cam


def main():
    # Header
    st.markdown(
        """
        <div style="background-color:#E8F1FF; padding: 28px; text-align: center; border-radius: 14px;">
            <h1 style="font-size: 2.4rem; margin-bottom: 0.4em; color:#222;">
                Accessible Object Explainer for Low-Vision Users
            </h1>
            <p style="font-size: 1.15rem; color:#333;">
                This demo uses a pretrained computer vision model and Grad-CAM explanations
                to help blind and low-vision users understand <b>what</b> is in an image,
                <b>where</b> it is located, and <b>why</b> the model made that prediction.
                You can also listen to an audio explanation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Step 1
    st.markdown(
        """
        <div style="background-color:#E8F1FF; padding: 18px; border-radius: 12px; margin-top: 1.8em;">
            <h2 style="font-size: 1.6rem; margin: 0; color:#222;">Step 1: Upload an image</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size: 1.15rem; color:#333; margin-top: 0.5em;'>"
        "Upload a photo (JPG or PNG). For example, a picture of a chair, desk, laptop, or everyday scene."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("There was a problem opening this image. Please try another file.")
        return

    # Step 2
    st.markdown(
        """
        <div style="background-color:#E8F1FF; padding: 18px; border-radius: 12px; margin-top: 2em;">
            <h2 style="font-size: 1.6rem; margin: 0; color:#222;">Step 2: Model analysis</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(image, use_container_width=True)

    with st.spinner("Analyzing image with ResNet18 and Grad-CAM..."):
        model, transform, class_names, grad_cam = get_model_and_helpers()

        label, prob, cam = analyze_image(
            image=image,
            model=model,
            transform=transform,
            class_names=class_names,
            grad_cam=grad_cam,
        )

        location_phrase = get_spatial_description(cam)
        spoken_location = location_phrase.replace("-", " ")
        overlay_img = overlay_heatmap_on_image(image, cam)

    st.markdown(
        "<h3 style='font-size: 1.4rem; margin-top:0.8em; color:#222;'>What the model sees:</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p style="font-size: 1.15rem; margin-bottom:2em; color:#333;">
            <b>Predicted object:</b> {label}<br>
            <b>Model confidence:</b> {prob:.1%}<br>
            <b>Location in the image:</b> {spoken_location}
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.image(overlay_img, caption="Grad-CAM explanation", use_container_width=True)

    # Step 3
    st.markdown(
        """
        <div style="background-color:#E8F1FF; padding: 18px; border-radius: 12px; margin-top: 1.2em;">
            <h2 style="font-size: 1.6rem; margin: 0; color:#222;">Step 3: Audio explanation</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation_text = (
        f"I think this image contains a {label}. "
        f"I am about {prob:.0%} confident. "
        f"The most important region is in the {spoken_location} area of the image. "
        "The highlighted colors show the pixels that most influenced my decision."
    )

    if (
        "last_explanation" not in st.session_state
        or st.session_state["last_explanation"] != explanation_text
    ):
        st.session_state["audio_bytes"] = generate_audio_bytes(explanation_text)
        st.session_state["last_explanation"] = explanation_text

    st.markdown(
        "<p style='font-size: 1.15rem; color:#333; margin-top: 0.5em;'>Press play to hear a spoken explanation.</p>",
        unsafe_allow_html=True,
    )

    audio_bytes = st.session_state.get("audio_bytes", None)
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Audio could not be generated. Please try re-running the app.")

    # Footer
    st.markdown(
        """
        <p style="text-align:center; font-size:0.9rem; color:#777; margin-top: 3em;">
            © 2025 Ailina Aniwan
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
