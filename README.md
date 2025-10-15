# Customer Churn Prediction with Advanced Feature Engineering

## 🚀 Project Overview

This project demonstrates the critical impact of **feature engineering** on the performance of a machine learning model. The primary goal is to predict customer churn for a telecommunications company by building and comparing two distinct models:
1.  A **Baseline Model** trained on raw, cleaned data.
2.  An **Enhanced Model** trained on a dataset enriched with custom-engineered features.

The project proves that thoughtful feature creation is often more impactful than model tuning alone for improving predictive accuracy and generating business insights.

---

## 🔧 Key Concepts & Technologies

* **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Core ML Concepts**:
    * Binary Classification (Logistic Regression, Random Forest)
    * Data Cleaning & Imputation
    * Advanced Feature Engineering (Binning, Interaction Features, Ratio Features)
    * Model Evaluation (Classification Report, F1-Score, Accuracy)
    * Feature Importance Analysis
* **Tools**: Jupyter Notebook, Git

---

## 📂 Dataset

The project utilizes the **Telco Customer Churn** dataset, which contains customer account information, demographic data, and the services they have signed up for. The notebook automatically clones the dataset from a public repository.

---

## 🛠️ Project Workflow

The notebook follows a structured, step-by-step approach to showcase the value of feature engineering:

1.  **Data Loading & Initial Exploration**: The dataset is loaded, and an initial inspection is performed to understand its structure and data types.
2.  **Data Cleaning & Preparation**: Critical preprocessing is performed, including:
    * Handling non-numeric values in the `TotalCharges` column.
    * Imputing missing values using the median.
    * Converting the target variable `Churn` to a binary format (0/1).
3.  **Baseline Model (Model 1)**: A `LogisticRegression` model is trained using a Scikit-learn `Pipeline` on the cleaned, original features. This establishes a benchmark performance metric (F1-score of ~0.60 for the churn class).
4.  **Advanced Feature Engineering**: Four new, intelligent features are created to provide deeper context to the model:
    * `tenure_group`: Bins the continuous `tenure` data into categorical groups (e.g., '0-1 Year').
    * `num_add_services`: Counts the total number of additional services a customer has.
    * Simplified categories for services to reduce noise.
    * `monthly_charge_ratio`: A ratio of `MonthlyCharges` to `tenure` to capture customer value/risk.
5.  **Enhanced Model (Model 2)**: The same `LogisticRegression` model and pipeline are trained on the new dataset containing the engineered features.
6.  **Performance Comparison & Analysis**: The performance of the enhanced model is compared against the baseline. A `RandomForestClassifier` is then used to analyze and visualize the **top 15 most important features**, confirming that the engineered features were highly influential in predicting churn.

---

## 📈 Results & Conclusion

The project successfully demonstrates that feature engineering has a tangible impact on model performance. While the overall accuracy saw a modest improvement from **~80% to ~81%**, the feature importance analysis confirmed that our custom-engineered features, such as `monthly_charge_ratio`, were among the top predictors of churn.

This serves as a practical blueprint for how to move beyond raw data to build more effective and insightful machine learning models.

---

## ▶️ How to Run

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-link>
    cd <your-repo-directory>
    ```
2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Run the Jupyter Notebook**:
    Launch Jupyter Notebook and open the `7_Preventing_Customer_Churn_with_Feature_Transformation.ipynb` file.
    ```bash
    jupyter notebook
    ```
