# house_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🏠 House Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File for House Prices", type="csv")

if uploaded_file is not None:
    df = pd.read_csv("houseprice.csv")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Drop missing values
    df.dropna(inplace=True)

    st.subheader("📊 Exploratory Data Analysis")

    # Show basic info
    st.write("Shape of dataset:", df.shape)
    st.write("Summary statistics:")
    st.write(df.describe())

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Select features and target
    st.subheader("🔧 Model Training")
    features = st.multiselect("Select Features", df.columns[:-1])
    target = st.selectbox("Select Target Variable", df.columns)

    if features and target:
        X = df[features]
        y = df[target]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        st.success("✅ Model Trained Successfully!")
        st.write("📈 R² Score:", round(r2_score(y_test, y_pred), 2))
        st.write("📉 RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

        # Prediction form
        st.subheader("📥 Make a Prediction")

        input_data = []
        for feature in features:
            val = st.number_input(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
            input_data.append(val)

        if st.button("Predict House Price"):
            pred_price = model.predict([input_data])[0]
            st.success(f"💰 Predicted House Price: {round(pred_price, 2)}")

else:
    st.info("👆 Please upload a CSV file to proceed.")
