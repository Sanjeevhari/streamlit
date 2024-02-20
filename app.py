# import module
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background
  
# Title
#set_background('./bgs/bg1.jpg')
st.title('Retinal Disease classification')

# set header
st.header('Please upload a retinal scan image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
model=load_model()