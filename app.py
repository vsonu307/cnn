import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

model = load_model('NN')

st.title('Image Classifier')
input_image = st.file_uploader('Drop the image')


if st.button('find'):
    image = load_img(input_image, target_size=(50, 50,3))
    img = img_to_array(image)
    img = np.array(img)
    img1 = img / 255.0
    img2 = img1.reshape(1, 50, 50, 3)
    a = model.predict(img2)
    if a<0.5:
        st.text('cat')
    else:
        st.text('dog')

    st.image(img1,width=500)
