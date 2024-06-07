import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Obesity Level Predictor')

gender = st.radio('Gender', ['Male', 'Female'])

age = st.number_input('Age', min_value=1)

height=st.number_input("Height",min_value=1.0)

Weight=st.number_input("Weight",min_value=1.0)

Family_History_With_Overweight = st.radio('Family History With Overweight', ['yes', 'no'])

Frequency_of_Consuming_High_Caloric_Food=st.radio('Frequency of Consuming High Caloric Food', ['yes', 'no'])
Frequency_of_Consuming_Vegetables=st.number_input('Frequency of Consuming Vegetables', min_value=1)

Number_of_Main_Meals_Per_Day=st.number_input('Number of Main Meals Per Day', min_value=1)

Consumption_of_Food_Between_Meals=st.radio('Consumption of Food Between Meals', ['Sometimes', 'Frequently', 'no', 'Always'])
Smoking_Habit=st.radio('Smoking Habit', ['yes', 'no'])
Daily_Consumption_of_Water=st.number_input('Daily Consumption of Water', min_value=1)
Monitoring_Caloric_Intake=st.radio('Monitoring Caloric Intake', ['yes', 'no'])
Frequency_of_Physical_Activity=st.number_input('Frequency of Physical Activity', min_value=1)
Time_Spent_Sitting_on_a_Weekday=st.number_input('Time Spent Sitting on a Weekday', min_value=1)
Consumption_of_Alcohol=st.radio('Consumption of Alcohol', ['Sometimes', 'no', 'Frequently'])
Mode_of_Transportation=st.radio('Mode of Transportation', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

bmi = Weight / (height / 100) ** 2
total_activity_score = Frequency_of_Physical_Activity * Time_Spent_Sitting_on_a_Weekday
water_intake_per_kg = Daily_Consumption_of_Water / Weight

if st.button('Predict Obesity Level'):
    # Prepare input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [Weight],
        'Family_History_With_Overweight': [Family_History_With_Overweight],
        'Frequency_of_Consuming_High_Caloric_Food': [Frequency_of_Consuming_High_Caloric_Food],
        'Frequency_of_Consuming_Vegetables': [Frequency_of_Consuming_Vegetables],
        'Number_of_Main_Meals_Per_Day': [Number_of_Main_Meals_Per_Day],
        'Consumption_of_Food_Between_Meals': [Consumption_of_Food_Between_Meals],
        'Smoking_Habit': [Smoking_Habit],
        'Daily_Consumption_of_Water': [Daily_Consumption_of_Water],
        'Monitoring_Caloric_Intake': [Monitoring_Caloric_Intake],
        'Frequency_of_Physical_Activity': [Frequency_of_Physical_Activity],
        'Time_Spent_Sitting_on_a_Weekday': [Time_Spent_Sitting_on_a_Weekday],
        'Consumption_of_Alcohol': [Consumption_of_Alcohol],
        'Mode_of_Transportation': [Mode_of_Transportation],
        'BMI': [bmi],
        'Total_Activity_Score': [total_activity_score],
        'Water_Intake_Per_Kg': [water_intake_per_kg]
    })

    # Load the pre-trained model
    with open("logistic_regression.pkl", 'rb') as file:
        pipeline = pickle.load(file)

    # Make prediction
    prediction = pipeline.predict(input_data)

    # Define a dictionary to map numerical output to label
    label_map = {
        0: 'Insufficient Weight',
        1: 'Normal Weight',
        2: 'Overweight Level I',
        3: 'Overweight Level II',
        4: 'Obesity Type I',
        5: 'Obesity Type II',
        6: 'Obesity Type III'
    }

    # Convert numerical prediction to label
    predicted_label = label_map[prediction[0]]

    # Display the predicted obesity level
    st.subheader('Predicted Obesity Level')
    st.write(predicted_label)
