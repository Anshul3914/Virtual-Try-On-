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

# import os
# import streamlit as st
# import torch
# import gdown
# import numpy as np
# from PIL import Image
# from datasets import VITONDataset
# from vition_test import VirtualTryOnTester
# from networks import GMM, ALIASGenerator
# from utils import load_checkpoint

# # Google Drive links for models
# MODEL_URLS = {
#     "seg.pth": "https://drive.google.com/uc?id=1sxKGOa-OAOKyUBDnYKfXIGJiRkCX55AM",
#     "gmm.pth": "https://drive.google.com/uc?id=1nUHGfNN9N8sbpj62H2Tc6_6w3nUpj5yy",
#     "alias.pth": "https://drive.google.com/uc?id=1AeBGmF1aBeDbdm5SAIMU-_38KtxfRGI4",
# }

# # Directory to store models
# CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# def download_models():
#     for model_name, url in MODEL_URLS.items():
#         model_path = os.path.join(CHECKPOINT_DIR, model_name)
#         if not os.path.exists(model_path):
#             st.info(f"Downloading {model_name}...")
#             gdown.download(url, model_path, quiet=False)
#         else:
#             st.success(f"{model_name} already exists.")

# def load_models():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     gmm = GMM()
#     alias = ALIASGenerator()
#     load_checkpoint(gmm, os.path.join(CHECKPOINT_DIR, "gmm.pth"))
#     load_checkpoint(alias, os.path.join(CHECKPOINT_DIR, "alias.pth"))
#     gmm.to(device).eval()
#     alias.to(device).eval()
#     return gmm, alias

# # Streamlit UI
# st.title("Virtual Try-On")
# st.write("Upload a person image and clothing image to generate a virtual try-on.")

# # File Upload
# person_image = st.file_uploader("Upload Person Image", type=["jpg", "png"])
# cloth_image = st.file_uploader("Upload Clothing Image", type=["jpg", "png"])

# if person_image and cloth_image:
#     st.image([person_image, cloth_image], caption=["Person", "Clothing"], width=200)
    
#     if st.button("Generate Try-On Image"):
#         st.info("Downloading models...")
#         download_models()
        
#         st.info("Loading models...")
#         gmm, alias = load_models()
        
#         st.info("Running Virtual Try-On...")
#         tester = VirtualTryOnTester(gmm, alias)
#         output_img = tester.run(person_image, cloth_image)
        
#         st.image(output_img, caption="Virtual Try-On Result", use_column_width=True)
#         st.success("Done!")
        
import os
import streamlit as st
import torch
import gdown
from PIL import Image
from datasets import VITONDataset
from networks import GMM, ALIASGenerator
from utils import save_images

# Try importing VirtualTryOnTester
try:
    from test import VirtualTryOnTester
except ImportError:
    st.error("Error: 'VirtualTryOnTester' not found in test.py. Ensure it's properly defined.")

# Google Drive links for pretrained models
MODEL_URLS = {
    "seg.pth": "https://drive.google.com/uc?id=1sxKGOa-OAOKyUBDnYKfXIGJiRkCX55AM",
    "gmm.pth": "https://drive.google.com/uc?id=1nUHGfNN9N8sbpj62H2Tc6_6w3nUpj5yy",
    "alias.pth": "https://drive.google.com/uc?id=1AeBGmF1aBeDbdm5SAIMU-_38KtxfRGI4",
}

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def download_models():
    """Download pretrained models if not available."""
    for model_name, url in MODEL_URLS.items():
        model_path = os.path.join(CHECKPOINT_DIR, model_name)
        if not os.path.exists(model_path):
            st.info(f"Downloading {model_name}...")
            gdown.download(url, model_path, quiet=False)
        else:
            st.success(f"{model_name} already exists.")

def load_models():
    """Load trained models for Virtual Try-On."""
    # Define model configuration (update according to your project)
    class Opt:
        def __init__(self):
            self.gpu_ids = []  # Empty list for CPU usage
            self.semantic_nc = 13  # Number of semantic channels (modify if needed)
    
    opt = Opt()
    inputA_nc = 3  # Number of channels for input A (RGB image)
    inputB_nc = 3  # Number of channels for input B (clothing image)

    # Initialize models with required parameters
    gmm = GMM(opt, inputA_nc, inputB_nc)
    alias = ALIASGenerator(opt, inputA_nc, inputB_nc)

    # Load weights onto the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmm.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gmm.pth"), map_location=device))
    alias.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "alias.pth"), map_location=device))

    gmm.eval()
    alias.eval()

    return gmm, alias

def main():
    st.title("👕 Virtual Try-On System")
    st.write("Upload a person image and clothing image to generate a try-on preview.")
    
    # Download models first
    download_models()
    
    # Check if VirtualTryOnTester is available
    if 'VirtualTryOnTester' not in globals():
        st.error("Error: VirtualTryOnTester is missing. Please check test.py.")
        return
    
    gmm, alias = load_models()
    
    # File uploader for person & clothing images
    person_img = st.file_uploader("Upload Person Image", type=["jpg", "png"])
    cloth_img = st.file_uploader("Upload Clothing Image", type=["jpg", "png"])
    
    if person_img and cloth_img:
        person = Image.open(person_img).convert("RGB")
        cloth = Image.open(cloth_img).convert("RGB")
        
        st.image([person, cloth], caption=["Person Image", "Clothing Image"], width=250)
        
        if st.button("Generate Try-On Result"):
            tester = VirtualTryOnTester(gmm, alias)
            result = tester.run(person, cloth)
            
            save_dir = "results"
            os.makedirs(save_dir, exist_ok=True)
            result_path = os.path.join(save_dir, "tryon_result.jpg")
            save_images([result], ["tryon_result.jpg"], save_dir)
            
            st.image(result_path, caption="Try-On Result", width=300)
            st.success("Try-On completed successfully!")

if __name__ == "__main__":
    main()


