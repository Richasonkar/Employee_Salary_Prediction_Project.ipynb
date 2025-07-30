# 🧠 Employee Salary Prediction Project

# This project aims to predict employee salaries based on features like experience, education, job role, and location using ML models.

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Step 2: Simulate Dataset ---
np.random.seed(42)
n_samples = 500

education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
education = np.random.choice(education_levels, size=n_samples)

experience = np.random.randint(0, 21, size=n_samples)

job_titles = ['Software Engineer', 'Data Scientist', 'HR Manager', 'Product Manager', 'Accountant']
job_title = np.random.choice(job_titles, size=n_samples)

locations = ['New York', 'San Francisco', 'Austin', 'Seattle', 'Chicago']
location = np.random.choice(locations, size=n_samples)

age = np.random.randint(22, 60, size=n_samples)

genders = ['Male', 'Female', 'Other']
gender = np.random.choice(genders, size=n_samples)

base_salary = 30000 + (experience * 2000) + (np.array([education_levels.index(e) for e in education]) * 5000)
base_salary = base_salary.astype(float)
base_salary += np.random.normal(0, 5000, size=n_samples)

df = pd.DataFrame({
    'YearsExperience': experience,
    'Education': education,
    'JobTitle': job_title,
    'Location': location,
    'Age': age,
    'Gender': gender,
    'Salary': base_salary.round(2)
})

print(df.head())

# --- Step 3: Preprocessing ---
X = df.drop('Salary', axis=1)
y = df['Salary']

categorical_cols = ['Education', 'JobTitle', 'Location', 'Gender']
numerical_cols = ['YearsExperience', 'Age']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

# --- Step 4: Model Training ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2 Score': r2_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T.round(2)

print("\nModel Performance Comparison:\n")                                       
print(results_df)
