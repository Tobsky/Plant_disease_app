import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from pathlib import Path


def set_background(image_file):




    return 0,0

#  Load classifier
@st.cache_resource
def load_model1():
    model1 = load_model(('New_efficientNetB5_V2.hdf5'), custom_objects={'KerasLayer':hub.KerasLayer})
    return model1

@st.cache_resource
def load_model2():
    model2 = load_model(('New_MobilenetV2_v3.hdf5'), custom_objects={'KerasLayer':hub.KerasLayer})
    return model2

@st.cache_resource
def load_model3():
    model3 = load_model(('PV_inceptionV3_v3.hdf5'), custom_objects={'KerasLayer':hub.KerasLayer})
    return model3

@st.cache_data
def classes():
    # Load classifier
    test_dir = 'C:/Users/Abdullateef/Python_programs/plant_images/test'
    test_dir = Path(test_dir)
    # Load class names
    CLASS_NAMES = np.array([item.name for item in test_dir.glob('*') if item.name != "LICENSE.txt"])
    return CLASS_NAMES


def classify(image, model, class_names, image_size):

    # Convert image
    image = image.resize(image_size)

    # Convert image to numpy array
    image_array = np.array(image).astype('float32')/255

    # set model input
    data = np.expand_dims(image_array, axis=0)

    # make predictions
    prediction = model.predict(data)
    score = tf.nn.softmax(prediction)
    index = np.argmax(score, axis=1)
    class_name = class_names[index]
    confidence_score = 100 * np.max(score)


    return class_name, confidence_score

def preprocess_image(image, target_size):
    resized_image = image.resize(target_size)
    image_array = np.array(resized_image).astype('float32') / 255.0
    return np.expand_dims(image_array, axis=0)

def classify2(image, model1, model2, model3, class_names, image_size_Eff, image_size_Mob, image_size_Incep):

    image_Eff = preprocess_image(image, image_size_Eff)
    image_Mob = preprocess_image(image, image_size_Mob)
    image_Incep = preprocess_image(image, image_size_Incep)

    All_preds = model_ensemble(image_Eff,image_Mob, image_Incep, model1, model2, model3)

    weights = [0.27182806, 0.56928691, 0.15888502]
    summed = np.tensordot(All_preds, weights, axes=((0),(0)))
    score = tf.nn.softmax(summed)
    result = np.argmax(summed, axis=1)

    class_name = class_names[result]
    confidence_score = 100 * np.max(score)

    return class_name, confidence_score


#Combine the all the models

def model_ensemble(image_Eff,image_Mob, image_Incep,model1, model2, model3,):
    models = [model1, model2, model3]
    for model in models:
        if model == model1:
            mnet_preds = model.predict(image_Eff)

        elif model == model2:
            eff_preds = model.predict(image_Mob)

        elif model == model3:
            inc_preds = model.predict(image_Incep)

    All_preds = [mnet_preds,eff_preds,inc_preds]  
    All_preds = np.array(All_preds)
    return All_preds