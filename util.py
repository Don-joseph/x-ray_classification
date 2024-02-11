from PIL import Image,ImageOps
import streamlit as st
import numpy as np
import base64
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    # Convert image size
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)

    # To numpy
    img_array = np.asarray(image) / 255.0
    data = np.ndarray(shape=(1, 150, 150, 3),dtype=np.float32)
    data[0] = img_array

    # Make prediction
    prediction = model.predict(data)
    print("ppp",prediction)
    x=prediction[0][0]
    print("xx",x)
    if x>0.8:
        i=1
    else:
        i=0
    print("iii",i)
    class_name = class_names[i]
    confi_score = prediction[0][0]

    return class_name, confi_score



