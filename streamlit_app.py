import streamlit as st
import torch
import asyncio
from options import get_opt  # Ensure this is correctly imported
from networks import SegGenerator  # Ensure correct imports

# Fix asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load model function
@st.cache_resource
def load_model():
    opt = get_opt()
    opt.init_type = getattr(opt, "init_type", "normal")
    opt.init_variance = getattr(opt, "init_variance", 0.02)
    
    model = SegGenerator(opt)
    model.eval()
    return model

st.title("👕 Virtual Try-On with VITON-HD")
st.markdown("Upload a person and clothing image to generate the try-on.")

# File upload
person_image = st.file_uploader("Upload person image", type=["png", "jpg", "jpeg"])
cloth_image = st.file_uploader("Upload clothing image", type=["png", "jpg", "jpeg"])

if person_image and cloth_image:
    model = load_model()
    with st.spinner("Processing..."):
        try:
            output_image = process_images(model, person_image, cloth_image)
            st.image(output_image, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating the try-on image: {str(e)}")

st.markdown("---")
st.markdown("👨‍💻 Developed using VITON-HD and Streamlit")
