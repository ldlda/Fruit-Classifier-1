import streamlit as st

st.set_page_config(page_title="Fruit Classifier - Home", layout="centered")

st.title("Fruit & Vegetable Classifier")

st.subheader(
    """
Welcome to the Fruit and Vegetable Classification project.

**Select a demo from the sidebar on the left** to get cooking.
"""
)

st.header("Project Features")
st.markdown(
    """
* **Run Prediction:** Upload an image and see predictions from both MobileNetV2 and EfficientNetV2-B0.
* **Grad-CAM:** See a heatmap of *why* the model made its prediction.
* **Real Time Demo:** See live predictions from the models.
* **Notebooks:** View the original Colab notebooks used to train the models.
"""
)
st.navigation(
    [
        st.Page("app.py", title="Home", icon="🏠"),
        st.Page("pages/prediction.py", title="Run Prediction", icon="🖼️"),
        st.Page("pages/grad_cam.py", title="Grad-CAM", icon="🔥"),
        st.Page("pages/realtime.py", title="Real-Time Demo", icon="🎥"),
        st.Page("pages/view_notebooks.py", title="Notebooks", icon="📓"),
    ],
    position="sidebar",
)
