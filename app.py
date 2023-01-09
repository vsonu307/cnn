import streamlit as st
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import numpy as np



pipe = pickle.load(open('model.pkl','rb'))

st.title('Image Classifier')
input_image = st.file_uploader('Drop the image')


if st.button('find'):
    image = load_img(input_image, target_size=(50, 50))
    img = img_to_array(image)
    img = np.array(img)
    img1 = img / 255.0
    img2 = img1.reshape(1, 50, 50, 3)
    a = pipe.predict(img2)
    if a<0.5:
        st.text('cat')
    else:
        st.text('dog')

    st.image(img1,width=500)
