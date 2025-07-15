import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

def generate_student_data(n_samples=200):
   np.random.seed(42)
   hours_studied = np.random.normal(5, 2, n_samples)
   hours_studied = np.clip(hours_studied, 0, 12)
   grade = hours_studied * 8 + 20 + np.random.normal(0, 10, n_samples)
   grade = np.clip(grade, 0, 100)
   return pd.DataFrame({'hours_studied': hours_studied, 'grade': grade})

#data generation and model training
df = generate_student_data()
X = df[['hours_studied']] 
y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

#streamlit app
st.title("🎓 Student Grade Predictor")

hours = st.number_input("Hours studied per week:", min_value=0.0, max_value=12.0, value=5.0)

if st.button("Predict Grade"):
   prediction = model.predict([[hours]])[0]
   st.success(f"Predicted Grade: {prediction:.1f}")