import streamlit as st
import torch
import os
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
# from test import run_viton_hd  # Assuming test.py runs VITON-HD inference
from viton_test import run_viton_hd


# Set page title
st.set_page_config(page_title="Virtual Try-On", layout="wide")

# Title and Description
st.title("👕 Virtual Try-On using VITON-HD")
st.write("Upload a person image and a clothing image to generate the try-on result.")

# Upload images
col1, col2 = st.columns(2)

with col1:
    person_image = st.file_uploader("Upload person image", type=["jpg", "png"])
    if person_image:
        st.image(person_image, caption="Person Image", use_column_width=True)

with col2:
    cloth_image = st.file_uploader("Upload clothing image", type=["jpg", "png"])
    if cloth_image:
        st.image(cloth_image, caption="Clothing Image", use_column_width=True)

# Run model when both images are uploaded
if person_image and cloth_image:
    if st.button("Generate Try-On"):
        with st.spinner("Processing... This may take a while ⏳"):
            # Save images to temporary directory
            person_path = f"data/test/person.jpg"
            cloth_path = f"data/test/cloth.jpg"
            
            Image.open(person_image).save(person_path)
            Image.open(cloth_image).save(cloth_path)
            
            # Run VITON-HD model (Assuming test.py has a function to handle this)
            output_image_path = run_viton_hd(person_path, cloth_path)  # Implement this in `test.py`
            
            # Display output
            if output_image_path and Path(output_image_path).exists():
                st.image(output_image_path, caption="Try-On Result", use_column_width=True)
            else:
                st.error("Error generating the try-on image. Check logs.")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed using VITON-HD and Streamlit")
