import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('models/resnet_finetuned_model.h5')

# Define class labels
class_labels = ['Leukemia', 'Myeloma', 'Normal'] 

# Detailed information about each class
class_info = {
    'Leukemia': {
        'description': 'Leukemia is a type of cancer found in your blood and bone marrow and is caused by the rapid production of abnormal white blood cells.',
        'image': 'static/sample_images/leuk.jpg'
    },
    'Myeloma': {
        'description': 'Myeloma is a type of blood cancer that affects plasma cells.',
        'image': 'static/sample_images/myelo.jpeg'
    },
    'Normal': {
        'description': 'No abnormalities detected, the blood cells appear normal.',
        'image': 'static/sample_images/norm.jpg'
    }
}


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/css/streamlit_style.css")


def preprocess_image(image):
    image = image.resize((128, 128))  
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array

def predict(image, model, class_labels):
    try:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]
        return class_labels[predicted_class], confidence
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        return None, None  # Indicates prediction failure


def model_description_page():
    st.title("Model Description")
    st.write("""
    ### ResNet Model
    The ResNet model (Residual Network) is a deep neural network that has been specifically fine-tuned for image classification tasks. This model is particularly effective at recognizing patterns in images due to its deep architecture and the use of residual learning. In this project, the ResNet model has been trained to classify images into three categories: Leukemia, Myeloma, and Normal.
    """)
    
    for label in class_labels:
        st.write(f"### {label}")
        st.write(class_info[label]['description'])
        st.image(class_info[label]['image'], use_column_width=True)
        st.write("---")


def prediction_page():
    st.title("Image Classification with ResNet")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        predicted_class, confidence = predict(image, model, class_labels)
        
        if predicted_class is not None:
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Information: {class_info[predicted_class]['description']}")
        else:
            st.write("Prediction failed. Please try again.")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Description", "Prediction"])

    if page == "Model Description":
        model_description_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == "__main__":
    main()
