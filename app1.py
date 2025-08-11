import streamlit as st
import numpy as np
from PIL import Image
import requests
import io
from tensorflow.keras.models import load_model

# إعدادات الصفحة
st.set_page_config(page_title="Car Damage Assessment", page_icon="🚗", layout="centered")

# إعدادات عامة
IMG_SIZE = (224, 224)
class_names = ['minor', 'moderate', 'severe']
model_path = "mobilenetv2_best.h5"  # تأكد إن الملف موجود في نفس مجلد التطبيق

# تحميل الموديل مع التخزين المؤقت
@st.cache_resource
def load_trained_model():
    return load_model(model_path)

model = load_trained_model()

# تحميل الصورة من رابط ثابت وتجهيزها
fixed_image_url = "https://imagesdrone.s3.eu-north-1.amazonaws.com/uploads/upload.jpg"

try:
    response = requests.get(fixed_image_url)
    response.raise_for_status()
    image_data = io.BytesIO(response.content)
    img = Image.open(image_data).resize(IMG_SIZE).convert('RGB')
except Exception as e:
    st.error(f"❌ Failed to load image: {e}")
    img = None

# عنوان التطبيق
st.title("🚗 Car Accident Damage Classification")
st.markdown("""
This app predicts the level of damage to a vehicle after an accident based on an image.

**Damage classes:**
- `minor` (light damage)
- `moderate` (medium damage)
- `severe` (heavy damage)
""")

if img:
    st.image(img, caption="Fixed Image from URL", use_column_width=True)

    # تجهيز الصورة للموديل
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = prediction[0][predicted_index]

        # عرض النتيجة مع تأثير لوني حسب التصنيف
        if predicted_label == "minor":
            st.success(f"✅ Predicted Damage: `{predicted_label}` (Light)")
        elif predicted_label == "moderate":
            st.warning(f"⚠️ Predicted Damage: `{predicted_label}` (Moderate)")
        elif predicted_label == "severe":
            st.error(f"🚨 Predicted Damage: `{predicted_label}` (Severe)")

        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        if st.button("🔁 Predict Again"):
            st.experimental_rerun()
else:
    st.error("No image to display or predict.")
 