import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

def add_sidebar():
    st.sidebar.write("🔧 Select the features of the flower to predict its type!")

    data = ("E:\\23881A66E2\\Projects\\Iris_Flower_Classification\\data\\IRIS.csv")


    sidebar_labels = {
        "sepal_length": "Length of your flowers Sepal",
        "sepal_width": "Width of your flowers Sepal",
        "petal_length": "Length of your flowers Petal",
        "petal_width": "Width of your flowers Petal"
    }

    sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 8.0, 5.0, step=0.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 5.0, 3.0, step=0.1)
    petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 7.0, 3.5, step=0.1)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 3.0, 1.0, step=0.1)

    return {
        "sepal_length" : sepal_length,
        "sepal_width" : sepal_width,
        "petal_length" : petal_length,
        "petal_width" : petal_width,
    }

def make_predictions(input_data):
    with open("E:\\23881A66E2\\Projects\\Iris_Flower_Classification\\model\\svm_model.pkl", "rb") as file:
        model = pickle.load(file)

    input_array = np.array([
        input_data["sepal_length"],
        input_data["sepal_width"],
        input_data["petal_length"],
        input_data["petal_width"]
    ]).reshape(1, -1)

    st.write("📥Your Input: ", input_array)

    prediction = model.predict(input_array)[0]
    probs = model.predict_proba(input_array)[0]
    return prediction, probs

def get_species_image(species_name):
    image_path = f"E:\\23881A66E2\\Projects\\Iris_Flower_Classification\\images\\{species_name.lower()}.jpg"
    return Image.open(image_path)

def main():
    st.set_page_config(
        page_title="Iris Flower Classification",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌸 Iris Flower Classification")
    st.write("This project is an interactive web application built using Streamlit that classifies Iris flowers into their respective species — Setosa, Versicolor, or Virginica — based on four morphological features: sepal length, sepal width, petal length, and petal width.")

    with st.expander("📂The dataset: "):
        data = pd.read_csv("E:\\23881A66E2\\Projects\\Iris_Flower_Classification\\data\\IRIS.csv")
        st.dataframe(data)

    with st.expander("📘 About the Iris Dataset"):
        st.markdown("""
        The Iris dataset is a classic dataset in pattern recognition and ML that includes 150 samples from 3 species: Setosa, Versicolor, Virginica and each sample includes 4 features:
        - Sepal Length
        - Sepal Width
        - Petal Length
        - Petal Width
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("images/setosa.jpg", caption="Setosa", width=200)
        with col2:
            st.image("images/versicolor.jpg", caption="Versicolor", width=200)
        with col3:
            st.image("images/virginica.jpg", caption="Virginica", width=200)

    input_data = add_sidebar()
    with st.container():
        prediction, probs = make_predictions(input_data)
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        species_name = species_map.get(prediction, "Unknown")

        col1, col2 = st.columns([2, 2])
        with col1:
            st.subheader("🔍 Predicted Species:")
            st.success(f"**{species_name}**")
            st.subheader("Prediction Probability:")
            st.bar_chart(probs)

            try:
                image = get_species_image(species_name)
                with col2:
                    st.subheader(":frame_with_picture: Image: ")
                    st.image(image, caption = f"{species_name} Flower", width=400)
            except FileNotFoundError:
                st.warning("Sorry, an error occured retrieving the image of the predicted flower!")



if __name__ == "__main__":
    main()