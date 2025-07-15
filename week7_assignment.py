# my app is availible to be accessed at https://arnavchopra-ct-csi-ds-4264.streamlit.app/
#simply copy paste this link in your browser and ypu can see what i have made!!
#and of course the source code is below


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def generate_student_data(n_samples=200):
   np.random.seed(42)
   hours_studied = np.random.normal(5, 2, n_samples)
   hours_studied = np.clip(hours_studied, 0, 12)
   grade = hours_studied * 8 + 20 + np.random.normal(0, 10, n_samples)
   grade = np.clip(grade, 0, 100)
   return pd.DataFrame({'hours_studied': hours_studied, 'grade': grade})

def get_letter_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

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
   letter_grade = get_letter_grade(prediction)
   st.success(f"Predicted Grade: {prediction:.1f} ({letter_grade})")
   
   #adding a small visualisation of grades
   st.subheader("📊 Grade Scale")
   
   grades = ['F', 'D', 'C', 'B', 'A']
   ranges = [30, 65, 75, 85, 95]
   colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']
   
   fig, ax = plt.subplots(figsize=(8, 4))
   bars = ax.bar(grades, ranges, color=colors)
   ax.axhline(y=prediction, color='red', linestyle='--', linewidth=2, label=f'Your Score: {prediction:.1f}')
   ax.set_ylabel('Score')
   ax.set_title('Grade Ranges (F: 0-59, D: 60-69, C: 70-79, B: 80-89, A: 90-100)')
   ax.legend()
   
   st.pyplot(fig)
