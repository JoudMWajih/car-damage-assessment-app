import streamlit as st
import numpy as np
from PIL import Image
import requests
import io
import os
from tensorflow.keras.models import load_model

# ✅ Set Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="Car Damage Assessment", page_icon="🚗", layout="centered")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Settings
IMG_SIZE = (224, 224)
class_names = ['minor', 'moderate', 'severe']
model_path = "mobilenetv2_best.h5"  # ✅ Define the model path

# Load the trained model (cached)
@st.cache_resource
def load_trained_model():
    return load_model(model_path)

model = load_trained_model()

# Function to process the image
def load_and_prepare_image(image_data, img_size=IMG_SIZE):
    img = Image.open(image_data).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# -----------------------
# 🎨 Streamlit UI
# -----------------------

st.title("🚗 Car Accident Damage Classification")
st.markdown("""
This app uses an AI model to evaluate the **level of damage** to a vehicle after an accident, based on an image.

🔍 **Possible classifications**:
- `minor` = Light damage
- `moderate` = Moderate damage
- `severe` = Severe damage

Upload an image or enter an image URL to get a prediction.
""")

# Choose input method
option = st.radio("Select image input method:", ["📁 Upload Image", "🌐 Image URL"])
image_data = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file

elif option == "🌐 Image URL":
    image_url = st.text_input("Enter image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image_data = io.BytesIO(response.content)
        except:
            st.error("❌ Failed to load image. Please check the URL.")

# Prediction
if image_data:
    img_array, img_display = load_and_prepare_image(image_data)
    st.image(img_display, caption='Selected Image', use_column_width=True)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = prediction[0][predicted_index]

        # Display result with visual cues
        if predicted_label == "minor":
            st.success(f"✅ Predicted Damage: `{predicted_label}` (Light)")
        elif predicted_label == "moderate":
            st.warning(f"⚠️ Predicted Damage: `{predicted_label}` (Moderate)")
        elif predicted_label == "severe":
            st.error(f"🚨 Predicted Damage: `{predicted_label}` (Severe)")

        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        # Restart button
        st.markdown("---")
        if st.button("🔁 Try Another Image"):
            st.experimental_rerun()
