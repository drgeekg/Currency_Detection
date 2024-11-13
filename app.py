import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import re
import requests
from collections import Counter

# Custom CSS for styling
st.markdown("""
    <style>
        /* Style for title */
        .title {
            font-size: 2em;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
            margin-top: -20px;
        }

        /* Style for button */
        .stButton > button {
            background-color: #4A90E2;
            color: white;
            font-size: 1em;
            padding: 0.5em 2em;
            border-radius: 8px;
        }

        /* Style for image captions */
        .image-caption {
            font-size: 0.9em;
            color: #333;
            text-align: center;
        }

        /* Style for section headers */
        .section-header {
            color: #4A90E2;
            font-weight: bold;
            font-size: 1.5em;
        }

        /* Style for input boxes */
        .stNumberInput input {
            border: 1px solid #4A90E2;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="title">Currency Detection and Conversion</div>', unsafe_allow_html=True)
st.write("Upload an image to detect currency notes using two YOLO models, verify detected classes, and convert detected values to another currency.")

# Load both YOLO models
@st.cache_resource
def load_models():
    model1 = YOLO("./model1.pt")
    model2 = YOLO("./model2.pt")
    return model1, model2

model1, model2 = load_models()

# Function to extract numeric value from class name
def extract_value(class_name):
    match = re.search(r"\d+", class_name)
    return int(match.group()) if match else 0

# Improved function to convert INR to selected currency with debug info
def convert_currency(amount_in_inr, target_currency):
    api_key = "d3020c8ec74c49b59be5df3fb47195ee"  # Replace with your actual API key
    url = f"https://openexchangerates.org/api/latest.json?app_id={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        rates = response.json().get("rates", {})
        inr_to_usd = rates.get("INR")
        target_rate = rates.get(target_currency)

        if inr_to_usd and target_rate:
            amount_in_usd = amount_in_inr / inr_to_usd
            converted_amount = amount_in_usd * target_rate
            return converted_amount
        else:
            st.error("Conversion rate not available for selected currency.")
            return None
    else:
        st.error("Failed to retrieve exchange rates. Check API key or network connection.")
        return None
    
if "final_total_value_inr" not in st.session_state:
    st.session_state.final_total_value_inr = None

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_array = np.array(image)
    
    # Model 1 Detection with loading spinner
    with st.spinner("Running Model 1..."):
        results1 = model1.predict(image_array)
        detected_image1 = results1[0].plot()
    st.image(detected_image1, caption="Model 1 Detection Results", use_column_width=True)

    model1_classes = [model1.names[int(result.cls)] for result in results1[0].boxes]
    model1_counts = Counter(model1_classes)

    # Model 2 Detection with loading spinner
    with st.spinner("Running Model 2..."):
        results2 = model2.predict(image_array)
        detected_image2 = results2[0].plot()
    st.image(detected_image2, caption="Model 2 Detection Results", use_column_width=True)

    model2_classes = [model2.names[int(result.cls)] for result in results2[0].boxes]
    model2_counts = Counter(model2_classes)

    # Verification Forms for both models
    st.write('<div class="section-header">Verification Forms</div>', unsafe_allow_html=True)

    possible_classes_model1 = ["1 Rupee", "2 Rupees", "5 Rupees", "10 Rupees", "20 Rupees", "50 Rupees", "100 Rupees", "200 Rupees", "500 Rupees", "2000 Rupees"]
    possible_classes_model2 = ["1 Rupee", "2 Rupees", "5 Rupees", "10 Rupees"]

    if "verified_model1_counts" not in st.session_state:
        st.session_state.verified_model1_counts = {cls: model1_counts.get(cls, 0) for cls in possible_classes_model1}
    if "verified_model2_counts" not in st.session_state:
        st.session_state.verified_model2_counts = {cls: model2_counts.get(cls, 0) for cls in possible_classes_model2}

    with st.form("verification_form"):
        st.write("### Model 1 Verification")
        for cls in possible_classes_model1:
            detected_count = model1_counts.get(cls, 0)
            st.session_state.verified_model1_counts[cls] = st.number_input(f"{cls} (Detected: {detected_count}) - Model 1", min_value=0, value=st.session_state.verified_model1_counts[cls])
        
        st.write("### Model 2 Verification")
        for cls in possible_classes_model2:
            detected_count = model2_counts.get(cls, 0)
            st.session_state.verified_model2_counts[cls] = st.number_input(f"{cls} (Detected: {detected_count}) - Model 2", min_value=0, value=st.session_state.verified_model2_counts[cls])
        
        confirm_counts = st.form_submit_button(label="Confirm Counts")

    if confirm_counts:
        total_value_model1 = sum(st.session_state.verified_model1_counts[cls] * extract_value(cls) for cls in st.session_state.verified_model1_counts)
        total_value_model2 = sum(st.session_state.verified_model2_counts[cls] * extract_value(cls) for cls in st.session_state.verified_model2_counts)
        
        st.session_state.final_total_value_inr = total_value_model1 + total_value_model2
        st.write(f"**Final Total Value Detected: {st.session_state.final_total_value_inr} INR**")

st.subheader("Convert Total Value to Another Currency")
target_currency = st.selectbox("Select target currency:", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD"])

if "converted_value" not in st.session_state:
    st.session_state.converted_value = None

if st.session_state.final_total_value_inr is not None and st.button("Convert"):
    with st.spinner("Converting currency..."):
        converted_value = convert_currency(st.session_state.final_total_value_inr, target_currency)
    if converted_value is not None:
        st.session_state.converted_value = converted_value
        st.write(f"**Converted Total Value: {converted_value:.2f} {target_currency}**")

if st.session_state.converted_value is not None:
    st.write(f"### Final Converted Value: {st.session_state.converted_value:.2f} {target_currency}")
