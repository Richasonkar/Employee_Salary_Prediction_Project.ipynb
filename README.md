  # 📊 Employee Salary Prediction – ML Project

This project aims to predict employee salaries using machine learning models based on features like experience, education, job title, and location. It is part of the Edunet Foundation & IBM SkillsBuild initiative to demonstrate practical applications of data science in HR analytics.

---

## 📁 Project Structure
Employee-Salary-Prediction/
├── Employee_Salary_Prediction_Project.ipynb
├── Employee_Salary_Prediction_PPT.pptx (optional)
├── Employee_Salary_Prediction_Report.pdf (optional)
├── requirements.txt
└── README.md


---
## 🧰 Tools & Technologies Used

- Python  
- Jupyter Notebook / Google Colab  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn

---

## 🔄 Project Workflow

1. **Data Generation** – A synthetic dataset was created with key employee features.
2. **Preprocessing** – Handled encoding, scaling, and cleaning.
3. **Model Training** – Applied Linear Regression, Random Forest, and XGBoost.
4. **Evaluation** – Compared models using MAE, RMSE, and R² Score.
5. **Result** – Linear Regression performed best (R² ≈ 0.89).
## 📊 Model Performance

| Model              | MAE (Rs) | RMSE (Rs) | R² Score |
|-------------------|----------|-----------|----------|
| Linear Regression | 4223.55  | 5081.84   | 0.89     |
| Random Forest     | 4644.34  | 5794.22   | 0.86     |
| XGBoost           | 5115.35  | 6275.09   | 0.84     |
## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Employee-Salary-Prediction.git
   cd Employee-Salary-Prediction
   Install dependencies:
pip install -r requirements.txt

Open the notebook:
jupyter notebook Employee_Salary_Prediction_Project.ipynb

 Conclusion
This project demonstrates the effectiveness of ML models in predicting employee salaries. It showcases how data science can enhance HR decision-making and bring transparency in compensation.



