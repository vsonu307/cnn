import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

model = load_model('NN',compile=False)
model.compile()

st.title('Dog/Cat Classification')
input_image = st.file_uploader('Drop the image')


if st.button('CHECK'):
    image = load_img(input_image, target_size=(50, 50,3))    
    img = img_to_array(image)
    img = np.array(img)
    img1 = img / 255.0
    img2 = img1.reshape(1, 50, 50, 3)
    a = model.predict(img2)
    if a<0.5:
        st.header('This seems CAT')
    else:
        st.header('This seems DOG')
    image1 = load_img(input_image)
    image1 = img_to_array(image1)
    image1 = np.array(image1)
    image1 = image1/255.0

    st.image(image1,width=500)
