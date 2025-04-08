# import streamlit as st
# import torch
# import numpy as np
# import gdown
# import os
# from PIL import Image
# from networks import SegGenerator, GMM, ALIASGenerator
# from utils import load_checkpoint, save_images
# from argparse import Namespace


# import asyncio

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())


# # Google Drive links for models
# MODEL_URLS = {
#     "seg.pth": "https://drive.google.com/uc?id=1sxKGOa-OAOKyUBDnYKfXIGJiRkCX55AM",
#     "gmm.pth": "https://drive.google.com/uc?id=1nUHGfNN9N8sbpj62H2Tc6_6w3nUpj5yy",
#     "alias.pth": "https://drive.google.com/uc?id=1AeBGmF1aBeDbdm5SAIMU-_38KtxfRGI4",
# }

# # Directory to store models
# CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Function to download models if not present
# def download_models():
#     for model_name, url in MODEL_URLS.items():
#         model_path = os.path.join(CHECKPOINT_DIR, model_name)
#         if not os.path.exists(model_path):
#             st.info(f"Downloading {model_name}...")
#             gdown.download(url, model_path, quiet=False)
#         else:
#             st.success(f"{model_name} already exists.")

# # Download models at runtime
# download_models()

# # Function to define model options
# from argparse import Namespace


# def get_opt():
#     return Namespace(
#         semantic_nc=13,  # Number of semantic classes
#         seg_checkpoint="checkpoints/seg.pth",  # Segmentation model checkpoint
#         gmm_checkpoint="checkpoints/gmm.pth",  # Geometric Matching Module checkpoint
#         alias_checkpoint="checkpoints/alias.pth",  # ALIAS generator checkpoint
#         input_nc=192,
#         output_nc=3,
#         init_type="normal",  # Initialization type for models
#         init_variance=0.02,  # Initialization variance
#         load_width=192,  # Input image width
#         load_height=256,  # Input image height
#         grid_size=5,  # Grid size for TPS transformation
#         num_upsampling_layers="more",  # ALIAS generator layers ("normal", "more", "most")
#         batch_size=1,  # Batch size for inference
#         dataroot="dataset",  # Dataset root
#         radius=5,  # Radius for pose visualization
#         workers=4,  # Number of worker threads
#         warp_feature=False,  # Whether to warp feature maps
#         use_dropout=False,  # Dropout in generator
#         norm_G="spectralaliasinstance",  # Generator normalization type
#         ngf=64  # ✅ Fixes the missing attribute issue
#     )



# def load_checkpoint(model, checkpoint_path):
#     if not os.path.exists(checkpoint_path):
#         raise ValueError(f"'{checkpoint_path}' is not a valid checkpoint path")
    
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint, strict=False)  # ✅ Ignores size mismatches



# # Function to load models
# def load_models(opt):
#     seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
#     gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
#     alias = ALIASGenerator(opt, input_nc=9)

#     load_checkpoint(seg, opt.seg_checkpoint)
#     load_checkpoint(gmm, opt.gmm_checkpoint)
#     load_checkpoint(alias, opt.alias_checkpoint)

#     seg.eval()
#     gmm.eval()
#     alias.eval()
#     return seg, gmm, alias

# # Function to process images
# def process_images(models, person_image, cloth_image):
#     seg, gmm, alias = models

#     # Convert images to tensors
#     person_tensor = torch.from_numpy(np.array(person_image)).permute(2, 0, 1).float() / 255
#     cloth_tensor = torch.from_numpy(np.array(cloth_image)).permute(2, 0, 1).float() / 255

#     person_tensor = person_tensor.unsqueeze(0)
#     cloth_tensor = cloth_tensor.unsqueeze(0)

#     # Perform inference
#     with torch.no_grad():
#         seg_output = seg(person_tensor)
#         gmm_output = gmm(cloth_tensor)
#         result_tensor = alias(seg_output, gmm_output)

#     # Convert tensor back to image
#     result_image = (result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     return Image.fromarray(result_image)

# # Streamlit Interface
# st.title("👕 Virtual Try-On")

# st.sidebar.header("Upload Images")
# person_file = st.sidebar.file_uploader("Upload Person Image", type=["jpg", "png"])
# cloth_file = st.sidebar.file_uploader("Upload Cloth Image", type=["jpg", "png"])

# if person_file and cloth_file:
#     person_image = Image.open(person_file).convert("RGB")
#     cloth_image = Image.open(cloth_file).convert("RGB")

#     st.image(person_image, caption="Person Image", use_container_width=True)
#     st.image(cloth_image, caption="Cloth Image", use_container_width=True)

#     if st.button("Try On"):
#         with st.spinner("Processing..."):
#             opt = get_opt()
#             models = load_models(opt)
#             result_image = process_images(models, person_image, cloth_image)
#             st.image(result_image, caption="Result", use_container_width=True)
#             st.success("Done!")

# # Footer
# st.markdown("---")
# st.markdown("👨‍💻 Developed using VITON-HD and Streamlit")


import os
import streamlit as st
import torch
import gdown
import cv2
import numpy as np
from PIL import Image
from test import main as run_vton

# Google Drive links for pretrained models
MODEL_URLS = {
    "seg.pth": "https://drive.google.com/uc?id=1sxKGOa-OAOKyUBDnYKfXIGJiRkCX55AM",
    "gmm.pth": "https://drive.google.com/uc?id=1nUHGfNN9N8sbpj62H2Tc6_6w3nUpj5yy",
    "alias.pth": "https://drive.google.com/uc?id=1AeBGmF1aBeDbdm5SAIMU-_38KtxfRGI4",
}

# Directory to store models
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Function to download models if not present
def download_models():
    for model_name, url in MODEL_URLS.items():
        model_path = os.path.join(CHECKPOINT_DIR, model_name)
        if not os.path.exists(model_path):
            st.info(f"Downloading {model_name}...")
            gdown.download(url, model_path, quiet=False)
        else:
            st.success(f"{model_name} already exists.")

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Streamlit UI
st.title("Virtual Try-On App")
st.write("Upload an image of a person and a clothing item to generate a virtual try-on result.")

# Download required models
download_models()

# Upload input images
person_image = st.file_uploader("Upload Person Image", type=["jpg", "png"])
cloth_image = st.file_uploader("Upload Clothing Image", type=["jpg", "png"])

if person_image and cloth_image:
    st.image([person_image, cloth_image], caption=["Person Image", "Clothing Image"], width=300)
    
    person_path = save_uploaded_file(person_image, "datasets/test/person.jpg")
    cloth_path = save_uploaded_file(cloth_image, "datasets/test/cloth.jpg")
    
    # Run virtual try-on model
    if st.button("Generate Try-On Image"):
        st.info("Processing... Please wait.")
        run_vton()  # Calls test.py
        
        # Show output
        result_path = "results/test/tryon.jpg"
        if os.path.exists(result_path):
            st.image(result_path, caption="Try-On Result", use_column_width=True)
        else:
            st.error("Failed to generate try-on image. Please check the logs.")

