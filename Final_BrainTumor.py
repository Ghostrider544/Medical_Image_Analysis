import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
import cv2
import base64
from tensorflow import image
import pyttsx3
import speech_recognition as sr

# Load the pre-trained model (Pickle format)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# VGG16 feature extractor (from Keras)
vgg16 = tf.keras.applications.VGG16(include_top=False, input_shape=(299, 299, 3))
preprocess_input = tf.keras.applications.vgg16.preprocess_input
image = tf.keras.preprocessing.image

# Batch size for processing
batch_size = 32

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Absolute path to Background image
bg_image_path = r"C:\\Users\\Amit\\Desktop\\Untitled Folder\Doctor.jpg"
# Get the base64 encoded image
encoded_image = get_base64_image(bg_image_path)

# Function to extract features using VGG16
def extract_features(img_paths, batch_size=batch_size):
    global vgg16
    n = len(img_paths)
    img_array = np.zeros((n, 299, 299, 3))

    for i, path in enumerate(img_paths):
        img = image.load_img(path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x = preprocess_input(img)
        img_array[i] = x

    X = vgg16.predict(img_array, batch_size=batch_size, verbose=1)
    X = X.reshape(n, 512, -1)
    return X

# Function to predict tumor type from image
def predict_tumor(img_path):
    X_test = extract_features([img_path])
    y_pred = model.predict(X_test)
    
    # Get the predicted class (index)
    predicted_class = np.argmax(y_pred[0])
    
    # Return the tumor type based on the predicted class
    if predicted_class == 2:
        return "Meningioma Tumor", "red"
    elif predicted_class == 0:
        return "Pituitary Tumor", "red"
    elif predicted_class == 1:
        return "No Tumor", "green"
    elif predicted_class == 3:
        return "Glioma Tumor", "red"
    else:
        return "Invalid input", "grey"

# Function to initialize text-to-speech engine
def speak(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

# Streamlit app interface
def app():
    # Title
    st.title("Brain Tumor Classification")

    # Brief explanation
    st.write("This application classifies brain tumor types based on uploaded MRI/CT images.")
    
    # Upload file
    uploaded_file = st.file_uploader("Upload a brain tumor image for prediction", type=["jpg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image temporarily
        img_path = os.path.join("temp_image.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image with a white border and reduced size
        st.markdown(
            f"""
            <style>
                .uploaded-image {{
                    border: 5px solid #FFFFFF;  /* White border */
                    padding: 5px;
                    border-radius: 10px;
                    max-width: 50%;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                }}
            </style>
            """, unsafe_allow_html=True
        )

        st.image(img_path, caption="Uploaded Image", use_container_width=False, width=500, output_format="JPEG")

        # Show a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(50)  # Progress halfway through

        # Predict the tumor type
        st.write("Classifying the uploaded image...")
        result, box_color = predict_tumor(img_path)

        # Complete the progress bar
        progress_bar.progress(100)

        # Display the result with a colored box
        st.markdown(
            f"""
            <div style="background-color:{box_color}; padding: 10px; border-radius: 5px; color: white; font-size: 18px;">
                Prediction: {result}
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Provide spoken feedback to the user
        speak(f"You have: {result}")

        # Optional: You can also show the image using OpenCV
        img = cv2.imread(img_path)
        st.image(img, channels="BGR", caption="Uploaded Image (Preview)", use_container_width=False, width=500)

        # Optional: Add a download button
        with open(img_path, "rb") as img_file:
            st.download_button(
                label="Download Image",
                data=img_file,
                file_name="classified_image.jpg",
                mime="image/jpeg"
            )

# Adding background image
st.markdown(
    f"""
    <style>
    .stApp {{
    background-image: url('data:image/jpeg;base64,{encoded_image}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    height: 100vh 100vw; }}
    </style>
    """, unsafe_allow_html=True
)

# Run the Streamlit app
if __name__ == "__main__":
    app()
