import streamlit as st
import asyncio
import torch
import numpy as np
from PIL import Image
from networks import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint, save_images
from argparse import Namespace

# Fix: Ensure a new event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Function to define model options
def get_opt():
    return Namespace(
        semantic_nc=13,  # Change according to your model
        seg_checkpoint="checkpoints/seg.pth",
        gmm_checkpoint="checkpoints/gmm.pth",
        alias_checkpoint="checkpoints/alias.pth",
    )

# Function to load models
def load_models(opt):
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    alias = ALIASGenerator(opt, input_nc=9)

    load_checkpoint(seg, opt.seg_checkpoint)
    load_checkpoint(gmm, opt.gmm_checkpoint)
    load_checkpoint(alias, opt.alias_checkpoint)

    seg.eval()
    gmm.eval()
    alias.eval()
    return seg, gmm, alias

# Function to process images
def process_images(models, person_image, cloth_image):
    seg, gmm, alias = models

    # Convert images to tensors (adjust preprocessing as per your model)
    person_tensor = torch.from_numpy(np.array(person_image)).permute(2, 0, 1).float() / 255
    cloth_tensor = torch.from_numpy(np.array(cloth_image)).permute(2, 0, 1).float() / 255

    person_tensor = person_tensor.unsqueeze(0)
    cloth_tensor = cloth_tensor.unsqueeze(0)

    # Perform inference (modify according to VITON-HD)
    result_tensor = alias(seg(person_tensor), gmm(cloth_tensor))

    # Convert tensor back to image
    result_image = (result_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    return Image.fromarray(result_image)

# Streamlit Interface
st.title("Virtual Try-On")

st.sidebar.header("Upload Images")
person_file = st.sidebar.file_uploader("Upload Person Image", type=["jpg", "png"])
cloth_file = st.sidebar.file_uploader("Upload Cloth Image", type=["jpg", "png"])

if person_file and cloth_file:
    person_image = Image.open(person_file)
    cloth_image = Image.open(cloth_file)

    st.image(person_image, caption="Person Image", use_container_width=True)
    st.image(cloth_image, caption="Cloth Image", use_container_width=True)

    if st.button("Try On"):
        with st.spinner("Processing..."):
            opt = get_opt()
            models = load_models(opt)
            result_image = process_images(models, person_image, cloth_image)
            st.image(result_image, caption="Result", use_container_width=True)
            st.success("Done!")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed using VITON-HD and Streamlit")
