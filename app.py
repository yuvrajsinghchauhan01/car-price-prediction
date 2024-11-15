import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define the app title
st.title('Car Price Predictor')

# Set up a container for the input fields
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        car_age = st.number_input('Age of the car')
        owner = st.number_input('Number of previous owners')
        distance = st.number_input('Distance traveled (in km)')
    with col2:
        car_name = st.selectbox('Car Name', [''] + list(pd.read_csv('cars_24_combined.csv')['Car Name'].unique()))
        fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
        location = st.selectbox('Location', [''] + list(pd.read_csv('cars_24_combined.csv')['Location'].unique()))
        drive = st.selectbox('Drive Type', ['Manual', 'Automatic'])
        car_type = st.selectbox('Car Type', ['HatchBack', 'Sedan', 'SUV', 'Lux_SUV', 'Lux_sedan'])

# Create a button to trigger prediction
with st.container():
    if st.button('Predict Price'):
        with st.spinner('Predicting...'):
            # Create a DataFrame with user input
            user_input = pd.DataFrame([[car_age, owner, distance, car_name, fuel_type, location, drive, car_type]],
                                       columns=['car_age', 'Owner', 'Distance', 'Car Name', 'Fuel', 'Location', 'Drive', 'Type'])

            # Preprocess the input data (similar to training data)
            categorical_cols = ['Car Name', 'Fuel', 'Location', 'Drive', 'Type']
            numerical_cols = ['car_age', 'Owner', 'Distance']

            # Create a preprocessing pipeline
            num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            cat_transformer = OneHotEncoder(handle_unknown='ignore')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, numerical_cols),
                    ('cat', cat_transformer, categorical_cols)
                ])

            # Assuming your training data is in a CSV named 'X.csv'
            # Load and fit the preprocessor on your training data
            df = pd.read_csv('X.csv')
            preprocessor.fit(df)

            # Transform the user input
            user_input_transformed = preprocessor.transform(user_input)

            # Make prediction
            predicted_price = model.predict(user_input_transformed)

            st.success(f'Predicted Car Price: ₹{predicted_price[0]}')