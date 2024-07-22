import streamlit as st
#from rembg import remove
#from PIL import Image
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"./keras_model.h5", compile=False)

# Load the labels
class_names = open(r"./labels.txt", "r").readlines()

#WEB GUI start
st.set_page_config(layout="centered", page_title="X-ray AI")

st.header("X-ray predictor", divider='rainbow')
st.sidebar.write("## Upload image :gear:")


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def fix_image(upload):
    with st.container(border=True):
        image = Image.open(upload)
        st.write("#Uploaded Image :camera:")
        st.image(image)

def predict(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    imagex = Image.open(img).convert("RGB")
    size = (224, 224)
    imagex = ImageOps.fit(imagex, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(imagex)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    st.write("__Class:__", class_name[1:])
    st.write("__Confidence Score:__", confidence_score)

def click(img):
    predict(img)
    fix_image(img)

my_upload = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])
st.sidebar.button(
    "__Predict x-ray__", on_click=predict, args=(my_upload,), disabled= my_upload is None
)
st.button("Use NORMAL example", on_click=click, args=("./NORMAL.jpeg",))
st.button("Use BACTERIA example", on_click=click, args=("./person1954_bacteria_4886.jpeg",))
st.button("Use VIRUS example", on_click=click, args=("./person563_virus_1103.jpeg",))

if my_upload is not None:
    fix_image(my_upload)