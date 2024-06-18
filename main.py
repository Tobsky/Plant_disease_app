import streamlit as st
import tensorflow as tf
import numpy as np
import folium
from streamlit_folium import st_folium
from PIL import Image
from util import classify, classify2, load_model1, load_model2, load_model3, classes

# Set page layout
st.set_page_config(
    page_title='Plant Disease Classification App',
    layout= 'wide',
    initial_sidebar_state='expanded'
)

data = []
data.append({'name': 'Location 1', 'latitude': float(8.851980687), 'longitude': float(7.742168711)})
data.append({'name': 'Location 2', 'latitude': float(9.072264), 'longitude': float(7.491302)})
data.append({'name': 'Base Staion', 'latitude': float(9.027665666), 'longitude': float(7.486341687)})
# data.append({'latitude': 9.082264, 'longitude': 7.691302})

# ABUJA_CENTER = (9.042506069, 7.399365484)
ABUJA_CENTER = (9.027665666, 7.486341687)
map = folium.Map(location=ABUJA_CENTER, zoom_start=12)

for station in data:
    location = (station['latitude'], station['longitude'])
    folium.Marker(location, popup=station['name']).add_to(map)

col1, col2, col3 = st.columns([0.5, 5, 0.5])  # Divide page into columns

# # Set title
col2.title('Plant Disease Recognition')

with col2:
    st_folium(map, width=1300, height=400)



st.sidebar.subheader("Input")  #  Set Sub-header

# List of models
models_list = ['EfficientNet', 'MobileNet', 'Inception', 'Differential Evolution']
selectEnsemble = st.sidebar.selectbox('Select a model', models_list)

# Upload an image
st.sidebar.subheader("Upload")
file = st.sidebar.file_uploader("Please upload a Plant Disease Image", type=["jpg", "jpeg", "png"])

# List of classes
CLASS_NAMES = classes()

#  Set classifier image size
image_size_Eff = (456,456)
image_size_Mob = (224, 224)
image_size_Incep = (299, 299)

# Load individual models
model1 = load_model1()
model2 = load_model2()
model3 = load_model3()


#  display Image
if file is not None:
    image = Image.open(file)
    col2.image(image, width = 500, use_column_width = True)

    if selectEnsemble == 'EfficientNet':
        #  Clasify Image
        class_name, conf_score = classify(image, model1, CLASS_NAMES, image_size_Eff)

        # write classification
        col2.success("#### {}".format(class_name))
        col2.success("#### Score for EfficientNet {:.2f}%".format(conf_score))
    
    elif selectEnsemble == 'MobileNet':
        #  Clasify Image
        class_name, conf_score = classify(image, model2, CLASS_NAMES, image_size_Mob)

        # write classification
        col2.success("## {}".format(class_name))
        col2.success("### score for MobileNet {:.2f}%".format(conf_score))

    elif selectEnsemble == 'Inception':
        #  Clasify Image
        class_name, conf_score = classify(image, model3, CLASS_NAMES, image_size_Incep)

        # write classification
        col2.success("## {}".format(class_name))
        col2.success("### score for Inception {:.2f}%".format(conf_score))

    elif selectEnsemble == 'Differential Evolution':
        #  Clasify Image
        class_name, conf_score = classify2(image, model1, model2, model3, CLASS_NAMES, image_size_Eff, image_size_Mob, image_size_Incep)

        # write classification
        col2.success("## {}".format(class_name))
        col2.success("### score for Differential Evolution {:.2f}%".format(conf_score))
else:
    col2.info("Awaiting the upload of the input image")