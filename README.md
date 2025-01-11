# Indian Currency Detection System

This project leverages machine learning and computer vision to create a real-time application for detecting and classifying Indian currency denominations. By utilizing the YOLOv8 object detection model, the system can identify both coins and banknotes while integrating real-time currency conversion using the Open Exchange Rates API. The user-friendly interface, developed with Streamlit, allows seamless interaction, ensuring accurate detection and conversion of Indian currency for applications in retail, travel, and finance.

## Features
- **Real-Time Currency Detection:** Identify various Indian currency denominations (coins and banknotes) with high accuracy.
- **Currency Conversion:** Convert detected values to other currencies using live exchange rates.
- **Human Verification:** Manually verify and correct detected values to ensure reliability.
- **Streamlit Interface:** Interactive UI for easy uploads, detection, and conversions.

## Highlights
- **Detection Accuracy:** 94% for coins and 91% for banknotes.
- **Real-Time Performance:** Average inference time of 0.08 seconds for coins and 0.1 seconds for banknotes.
- **Future Expansion:** Plans to include multi-currency support, offline conversions, and mobile deployment.

## How It Works
1. Upload an image of Indian currency (coins or notes).
2. The application detects and classifies denominations using YOLOv8 models.
3. Verify or correct the detected results for accuracy.
4. Convert the total INR value to other currencies like USD, EUR, or GBP in real time.

## Installation
1. Clone the repository: `git clone https://github.com/drgeekg/Currency_Detection`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Future Enhancements
- Support for additional currencies.
- Offline conversion capabilities.
- Mobile application development for on-the-go usability.
