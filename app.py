import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide",
    page_icon="🌿"
)
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Use your actual image name
add_bg_from_local("background.jpg")

# Load the model
model = load_model("plant_disease_model.h5")

# Class labels (same as your training classes)
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Sidebar
st.sidebar.title("🧬 About this App")
st.sidebar.markdown("""
## 🧬 About This App

- Detects plant diseases from leaf images  
- Upload an image → get prediction + confidence  
- Built for farmers, students & researchers  
- Fast, accurate & easy to use  

---

## 🧠 Model Info

- Based on MobileNetV2 (Transfer Learning)  
- Trained on 50,000+ leaf images  
- Covers 38 plant disease categories  
- Input size: 224 × 224  
- Accuracy: ~90% on validation set  

---

## 📂 Dataset

- Source: Kaggle  
- Name: New Plant Diseases Dataset (Augmented)  
- Link: [View on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  

---

## 👨‍💻 Developer

- Abhishek,Tanya,Tauhid,Nakshtra
- B.Tech Final Year  
- Capstone Project   
""")


# Main Title
st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload a leaf image and detect the disease in seconds!</h4>", unsafe_allow_html=True)
st.write("")

# File uploader
file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if file:
    # 1. Load & preprocess the image
    img = Image.open(file).convert("RGB")
    resized_img = img.resize((224, 224))
    img_array = image.img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Make prediction
    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds)

    # 3. Display uploaded image and top prediction
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    with col2:
        st.markdown("### 🧠 Prediction")
        st.success(f"Detected: {pred_class}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
        st.progress(int(confidence * 100))

    # 4. Show confidence graph for all classes
    import pandas as pd
    import matplotlib.pyplot as plt

    pred_df = pd.DataFrame(preds[0], index=class_labels, columns=["Confidence"])
    pred_df = pred_df.sort_values(by="Confidence", ascending=True)

    st.markdown("### 📊 Prediction Confidence Across All Classes")

    fig, ax = plt.subplots(figsize=(6, len(class_labels) // 2))
    pred_df.plot.barh(ax=ax, legend=False, color='teal')
    ax.set_xlabel("Confidence Score")
    ax.set_xlim([0, 1])
    st.pyplot(fig)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ by <b>Abhishek,Tanya,Tauhid,Nakshtra</b></div>",
    unsafe_allow_html=True
)
