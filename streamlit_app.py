import streamlit as st
from PIL import Image
import os

st.title("👗 Virtual Try-On Demo")

person_image = st.file_uploader("Upload Person Image", type=["jpg", "jpeg", "png"])
cloth_image = st.file_uploader("Upload Cloth Image", type=["jpg", "jpeg", "png"])

if st.button("Try-On") and person_image and cloth_image:
    os.makedirs("datasets/test/image", exist_ok=True)
    os.makedirs("datasets/test/cloth", exist_ok=True)
    
    person_path = "datasets/test/image/person.jpg"
    cloth_path = "datasets/test/cloth/cloth.jpg"
    
    with open(person_path, "wb") as f:
        f.write(person_image.read())
    with open(cloth_path, "wb") as f:
        f.write(cloth_image.read())

    with open("datasets/test/test_pairs.txt", "w") as f:
        f.write("person.jpg cloth.jpg\n")

    st.text("Processing...")
    os.system("python test.py")

    result_path = "results/test/person.jpg"
    if os.path.exists(result_path):
        result = Image.open(result_path)
        st.image(result, caption="👕 Final Output")
    else:
        st.error("Result image not found.")
