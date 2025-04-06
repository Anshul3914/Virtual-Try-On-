import os
import gdown

# Google Drive file IDs (Extracted from your links)
model_urls = {
    "alias_final.pth": "1AeBGmF1aBeDbdm5SAIMU-_38KtxfRGI4",
    "seg_final.pth": "1sxKGOa-OAOKyUBDnYKfXIGJiRkCX55AM",
    "gmm_final.pth": "1nUHGfNN9N8sbpj62H2Tc6_6w3nUpj5yy",
}

# Create checkpoints directory if not exists
os.makedirs("checkpoints", exist_ok=True)

# Download each model file
for filename, file_id in model_urls.items():
    output_path = f"checkpoints/{filename}"
    
    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    else:
        print(f"{filename} already exists, skipping.")

print("All models downloaded successfully!")
