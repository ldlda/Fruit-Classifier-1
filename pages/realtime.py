import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

from utils import MODEL_CONFIG, FruitClassifierProcessor

st.set_page_config(page_title="Real-Time Demo", layout="centered")
st.title("Real-Time Classifier")
st.write("Select a model and click 'START' to activate your webcam.")

# --- MODEL SELECTION ---
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "MobileNetV2"

_options = list(MODEL_CONFIG.keys())
_default_index = _options.index(st.session_state["model_choice"])

st.selectbox(
    "Choose a model",
    options=_options,
    index=_default_index,
    key="model_choice",
)

# --- RUN THE WEBRTC STREAMER ---
webrtc_streamer(
    key="realtime_classifier",
    video_processor_factory=FruitClassifierProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

st.caption(
    f"Using {st.session_state['model_choice']} model. Change model and restart webcam."
)
