import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer




# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"c:\Users\sanja\OneDrive\Desktop\Salary Data.csv")

    # Fill missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Education Level'].fillna(df['Education Level'].mode()[0], inplace=True)
    df['Salary'].fillna(df['Salary'].median(), inplace=True)
    df['Job Title'].fillna(df['Job Title'].mode()[0], inplace=True)
    df['Years of Experience'].fillna(df['Years of Experience'].median(), inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Encode Gender
    df['Gender'] = df['Gender'].map({"Male": 0, "Female": 1})

    # One Hot Encoding for Education Level
    df = pd.get_dummies(df, columns=['Education Level'], drop_first=True)

    return df

df = load_data()

# -------------------------------
# Feature Selection
# -------------------------------
X = df.drop(['Salary', 'Job Title'], axis=1)
y = df['Salary']

# Scaling
scaler = StandardScaler()
X[['Age', 'Years of Experience']] = scaler.fit_transform(
    X[['Age', 'Years of Experience']]
)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("IT Salary Predictor")
st.write("Enter employee details to predict salary")

age = st.number_input("Age", min_value=22, max_value=65, value=25)
experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=1)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

# Manual Encoding
gender_val = 0 if gender == "Male" else 1

edu_bachelors = 1 if education == "Bachelor's" else 0
edu_masters = 1 if education == "Master's" else 0
edu_phd = 1 if education == "PhD" else 0

# Scale numeric input
scaled_values = scaler.transform([[age, experience]])
age_scaled = scaled_values[0][0]
exp_scaled = scaled_values[0][1]

# Create Input DataFrame
input_data = pd.DataFrame([[
    age_scaled,
    exp_scaled,
    gender_val,
    edu_masters,
    edu_phd
]], columns=X.columns)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹ {round(prediction, 2)}")
