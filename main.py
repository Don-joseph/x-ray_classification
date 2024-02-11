import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify,set_background
set_background("C:\\Users\\donjo\\Downloads\\R.jpg")
from PIL import ImageOps
st.title("pneumonia classification")
st.header("please upload a chest x-ray")
file=st.file_uploader('',type=['jpeg','jpg','png'])
model=load_model("C:\\Users\\donjo\\Downloads\\pneu_model.h5")
with open("C:\\Users\\donjo\\Downloads\\labels.txt",'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
print(class_names)

#display img
if file is not None:
    image=Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)
    # classify the img
    class_name, conf_score=classify(image,model,class_names)
    #write classi
    st.write("## {}".format(class_name))
    st.write("### score {}".format(conf_score))
