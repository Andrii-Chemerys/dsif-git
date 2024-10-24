

api_url = "http://localhost:8503"

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.title("Fraud Detection App")

# Display site header
#image = Image.open("../images/dsif header.jpeg")

image_path = "../images/dsif header 2.jpeg"
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_column_width=True)  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

data = {
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "customer_balance": customer_balance
    }

if st.button("Show Feature Importance"):
    # import matplotlib.pyplot as plt
    response = requests.get(f"{api_url}/feature-importance")
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

if st.button("Predict and show prediction confidence"):
    # Make the API call

    response = requests.post(f"{api_url}/predict/",
                            json=data)
    result = response.json()
    confidence = result['confidence']

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    # Confidence Interval Visualization
    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

if st.button("Predict and show SHAP values"):
    response = requests.post(f"{api_url}/predict/",
                             json=data)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    ######### SHAP #########
    # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("SHAP Values Explanation")

    # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    st.pyplot(fig)

# SOLUTION
# by Andrii Chemerys
#################################################################
# Exercise 2.1:                                                 #
# Adding a File Upload Section and Saving Predictions in CSV    #
#################################################################
import pandas as pd
from tkinter import *
from tkinter.filedialog import askdirectory

# Add field on streamlit for uploading csv files
upload_file = st.file_uploader("Upload file", type=['csv'])

if upload_file:
    # When file is uploaded process it to DataFrame
    transactions = pd.read_csv(upload_file)
    transactions.set_index(transactions.columns[0], inplace=True)
    # Check does it contain 'must have' columns
    must_have_cols = set(['transaction_amount', 'transaction_time', 'customer_age', 'customer_balance'])
    csv_struct = set(transactions.columns.to_list())
    missing_cols = must_have_cols - csv_struct
    if len(missing_cols):
        # Inform user if some of 'must have' columns are missing
        st.error(f"File must have {must_have_cols} columns.\nMissing columns: '{missing_cols}'", icon=":material/block:")
    else:
        # Otherwise make dictionary from DataFrame
        # and post it to the server for processing
        payload = transactions.to_dict(orient='split')
        response = requests.post(f"{api_url}/uploadfile/", json=payload)
        # Reconstruct received from server json object to DataFrame
        dict_df = response.json()['json']
        transactions_pred = pd.DataFrame(
            data=dict_df['data'],
            index=dict_df['index'],
            columns=dict_df['columns']
        )
        # Show it to the user
        st.caption("Result of fraud predictions")
        st.dataframe(transactions_pred, selection_mode='single-row')

        # Allow to save results in csv file
        st.download_button("Download result",
                           data=transactions_pred.to_csv(),
                           file_name=upload_file.name[:-4]+"_results.csv",
                           mime="text/csv")

        # Alternative way to save file in custom directory
        ## In my environment call to askdirectory terminates streamlit with a message:
        ### Terminating app due to uncaught exception 'NSInternalInconsistencyException',
        ### reason: 'NSWindow drag regions should only be invalidated on the Main Thread!'
        # if st.button("Download result to folder"):
        #     path = rf"{askdirectory(title='Choose folder')}/"
        #     transactions_pred.to_csv(path+upload_file.name[:-4]+"_results.csv")


#################################################################
# Exercise 2.2:                                                 #
# Adding Visuals to Streamlit App                               #
#################################################################

        # Add new features
        transactions_pred['transaction_amount_to_balance_ratio'] = transactions_pred.transaction_amount/transactions_pred.customer_balance
        transactions_pred['colors'] = np.where(transactions_pred.pred_fraud == 1, '#ff0000', '#00ff00')
        # Features' selection to plot
        st.caption("Select pair of dataset's columns for scatter plot")
        x_axis = st.selectbox("X axis", options=transactions_pred.select_dtypes('number').columns)
        y_axis = st.selectbox("Y axis", options=transactions_pred.select_dtypes('number').drop(x_axis, axis=1).columns)

        # Plot the chart
        st.scatter_chart(transactions_pred, x=x_axis, y=y_axis, color='colors')
