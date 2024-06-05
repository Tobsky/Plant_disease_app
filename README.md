# Plant Disease Classification App

This repository contains a Streamlit web application for classifying plant diseases using deep learning models. 
The app allows users to upload an image of a plant leaf and select from multiple pre-trained models to classify 
the disease present in the leaf.
Each pre-trained model was trained using transfer learning on the plant village dataset and an ensemble technique called differential evolution was used to combine the models to get a better prediction and also to make the predictions stable.

## Features

- **Interactive Map:** Displays locations on a map using Folium.
- **Model Selection:** Choose between EfficientNet, MobileNet, Inception, and Differential Evolution for classification.
- **Image Upload:** Upload a plant disease image for classification.
- **Prediction Display:** Shows the predicted class and confidence score.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tobsky/Plant_disease_app.git

2. Navigate to the project directory:
    ```bash
    cd plant-disease-classification

3. Install the required packages:
    ```bash
    pip install -r requirements.txt

## Usage

Run the Streamlit app:
    ```bash
    streamlit run main.py

## Models
The following models are used in this app:

- **EfficientNet**
- **MobileNet**
- **Inception**
- **Differential Evolution**

Each model can be selected from the sidebar in the app for classification.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.