# Penguin Species Classification with Random Forest

This project builds a machine learning model to classify penguin species using a Random Forest Classifier. The model is trained on a dataset of penguin physical measurements and attributes, and the results are visualized and deployed using a Streamlit web application. The project draws inspiration from *Streamlit for Data Science* by Andrien Treville, which provided guidance on structuring the code and deploying it with Streamlit.

## Project Overview

The goal of this project is to predict the species of penguins (Adélie, Chinstrap, or Gentoo) based on features such as island, bill length, bill depth, flipper length, body mass, and sex. The Random Forest Classifier is trained on cleaned data, and its performance is evaluated using accuracy. Feature importance is visualized to understand which attributes contribute most to species prediction. The trained model is then deployed as an interactive Streamlit app.

Live app: [Penguin Species Prediction](https://joshuasalamipeter-penguins-machine-learning.streamlit.app/)

## Dataset

The dataset used is `penguins.csv`, which contains measurements of penguins from three species. Key columns include:
- `species`: Target variable (Adélie, Chinstrap, Gentoo)
- `island`: Island where the penguin was observed
- `bill_length_mm`: Length of the bill in millimeters
- `bill_depth_mm`: Depth of the bill in millimeters
- `flipper_length_mm`: Length of the flipper in millimeters
- `body_mass_g`: Body mass in grams
- `sex`: Sex of the penguin (male or female)

## Requirements

To run this project locally, install the following Python libraries with the specified versions:
- `altair==5.5.0`
- `matplotlib==3.10.1`
- `pandas==2.2.3`
- `scikit-learn==1.6.1`
- `seaborn==0.13.2`
- `streamlit==1.43.2`

You can install them using pip by creating a `requirements.txt` file with the above versions and running:
```bash
pip install -r requirements.txt