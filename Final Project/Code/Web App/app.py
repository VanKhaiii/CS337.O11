import streamlit as st
import streamlit.components.v1 as components
import base64
import pandas as pd
import numpy as np
import os
import sys
#sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from utils.py, config.py
from utils import * 
from config import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from app_helpers import * 
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="VietNam Traditional classification",
    page_icon="🎞🎞🎞🎞",
    layout="wide",
    initial_sidebar_state="expanded",
)

model1 = tf.keras.models.load_model("checkpoint\\best_model1.h5", compile="false")
model1.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=[METRICS[0]])

model2 = tf.keras.models.load_model("checkpoint\\best_model2.h5", compile="false")
model2.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=[METRICS[0]])

model3 = tf.keras.models.load_model("checkpoint\\best_model3_ver2.h5", compile="false")
model3.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=[METRICS[0]])

def process_audio_user(uploaded_files):
    """
    Process 
    Input: upload_files (multiples): The UploadedFile class get from st.file_uploader

    Create and save audio uploaded into audio_from_user folder

    Output:
    - audio_list: List of audio directories from user as argument for `predict_new()` 
    """
    audio_list = []
    if uploaded_files is not None:
        for i, file in enumerate(uploaded_files):
            if file.name[-4:] != ".wav" and file.name[-4:] != ".mp3":
                st.error("Sorry! Please upload audio with .wav or .mp3 extension.")
                break
            else:
                if not os.path.exists(AUDIO_FROM_USER):
                    os.makedirs(AUDIO_FROM_USER)
                    print("Created audio_from_user folder")
                st.write("File uploaded {}:".format(i), file.name)
                st.audio(file)
                # Save file to audio_from_user folder
                with open(os.path.join(AUDIO_FROM_USER, file.name), "wb") as f:
                    f.write(file.getbuffer())
                audio_dir = os.path.join(AUDIO_FROM_USER, file.name)
                # Save audio directory for prediction
                audio_list.append(audio_dir)
                
    return audio_list
    
def main():
    st.markdown(
        title_style,
        unsafe_allow_html=True,
    )
    
    st.markdown(result_style, unsafe_allow_html=True)
    
    title = """
        <h1 class = "title" >Vietnamese Traditional Music Classifier</h1>
        </div>
    """
    
    request = """
        <h2 class = "request">Upload your audio below. (.wav, .mp3) </h2>
        </div>
    """
    
    endding = """
        <h2 class = "result">Prediction Completed!</h2>
        </div>
    """
    
    st.markdown(title,
                unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.text("")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    image_width = 200
    image_height = 100

    with col1:
        image = Image.open("cailuong.jpg")
        new_image = image.resize((400, 300))
        st.image(new_image, width=image_width , caption="Cải lương", use_column_width="auto")

    with col2:
        image = Image.open("catru.jpg")
        new_image = image.resize((400, 300))
        st.image(new_image, caption="Ca trù", use_column_width="auto")

    with col3:
        image = Image.open("chauvan.jpg")
        new_image = image.resize((400, 300))
        st.image(new_image, caption="Chầu văn", use_column_width="auto")

    with col4:
        image = Image.open("cheo.jpg")
        new_image = image.resize((400, 300))
        st.image(new_image, caption="Chèo", use_column_width="auto")

    with col5:
        image = Image.open("hatxam.jpg")
        new_image = image.resize((400, 300))
        st.image(new_image,caption="Hát xẩm", use_column_width="auto")
        
    st.text("")
    st.markdown(request, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp3", "wav"], accept_multiple_files = True,)                  
    audio_dirs = process_audio_user(uploaded_files=uploaded_file)
    if st.button("Classify"):
        
        filenames = [file.name for file in uploaded_file]

        y_pred, y_class = PROD_predict(audio_dir = audio_dirs, src_folder = AUDIO_FROM_USER, model1=model1, model2=model2, model3 = model3,  save_dir=FOLDER_ROOT)

        for i, file in zip(range(0, len(y_pred)), audio_dirs):
            st.success("File uploaded {}: {} --> PREDICT: {}".format(i, file.split("\\")[-1], y_class[i]))
        st.markdown(endding, unsafe_allow_html=True)

        list_data = pd.read_excel(EXCEL_URL, index_col=0)

        new_input = create_data_input(filename_list=filenames, label_list=y_class, list_data=list_data, index_previous=len(list_data) - 1)

        save_prediction(EXCEL_URL, list_data, new_input)

if __name__=='__main__':
    set_background('blur.jpg')
    main()