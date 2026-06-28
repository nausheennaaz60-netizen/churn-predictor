# Customer Churn Risk Predictor
A machine learning web app that predicts customer churn probability 
in real time using XGBoost, with SHAP explainability and Supabase authentication.

## Live Demo
🔗 https://churn-predictor-5ggcxdbwktvenhdji4hhjx.streamlit.app

## Try This Demo
After logging in, enter this profile to see the model in action:

| Field | Value |
| Tenure - 5 months |
| Contract - Month-to-month |
| Internet - Fiber optic |
| Monthly charges - $95 |
| Tech support - No |

**Expected result: Medium-to-High churn risk (approximately 60–75%, depending on the trained model).

## Key Insight
The model identifies **contract type, monthly charges, and customer tenure** as the strongest drivers of churn.
Customers with **month-to-month contracts**, **higher monthly charges**, and **shorter tenure** are consistently predicted to have a higher churn risk.

**Business action:** Focus retention campaigns on these high-risk customers by offering personalized discounts, loyalty rewards, 
 or incentives to switch to long-term contracts.
## Features

- Signup / Login with Supabase authentication
- Real-time churn probability prediction (XGBoost, 81% accuracy)
- SHAP explainability — shows WHY each customer is at risk
- Retention recommendations based on risk profile
- Built with Python, Streamlit, XGBoost, SHAP, Supabase

## Tech Stack
- **ML Model:** XGBoost trained on IBM Telco Churn dataset (7,032 rows)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Frontend:** Streamlit
- **Auth & Database:** Supabase (PostgreSQL)
- **Deployment:** Streamlit Community Cloud

## How to run locally
pip install -r requirements.txt
streamlit run app.py 
