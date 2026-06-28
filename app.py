import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from supabase import create_client, Client

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📊", layout="wide")

# ── SUPABASE CLIENT ───────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

supabase = get_supabase()

# ── SESSION STATE INIT ────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"

# ── AUTH FUNCTIONS ────────────────────────────────────────────────
def sign_up(email, password):
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        if res.user:
            return True, "Account created! You can now log in."
        return False, "Signup failed. Try again."
    except Exception as e:
        return False, str(e)

def sign_in(email, password):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            st.session_state.user = res.user
            return True, "Logged in successfully"
        return False, "Invalid credentials"
    except Exception as e:
        return False, "Invalid email or password"

def sign_out():
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

# ── AUTH UI ───────────────────────────────────────────────────────
if not st.session_state.user:
    st.title("Customer Churn Risk Predictor")
    st.caption("Sign in to access the prediction tool.")
    st.divider()

    col_left, col_mid, col_right = st.columns([1, 1.2, 1])
    with col_mid:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            st.subheader("Welcome back")
            login_email    = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_pw")

            if st.button("Login", type="primary", use_container_width=True):
                if login_email and login_password:
                    ok, msg = sign_in(login_email, login_password)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in both fields")

        with tab_signup:
            st.subheader("Create account")
            signup_email    = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password (min 6 chars)",
                                            type="password", key="signup_pw")
            signup_confirm  = st.text_input("Confirm password",
                                            type="password", key="signup_confirm")

            if st.button("Create Account", type="primary", use_container_width=True):
                if not signup_email or not signup_password:
                    st.warning("Please fill in all fields")
                elif signup_password != signup_confirm:
                    st.error("Passwords don't match")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    ok, msg = sign_up(signup_email, signup_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
    st.stop()

# ── LOGGED IN — SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.success("Logged in")
    st.write(f"**{st.session_state.user.email}**")
    st.divider()
    if st.button("Logout", use_container_width=True):
        sign_out()

# ── LOAD MODEL ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

artifact = load_model()
model    = artifact["model"]

# ── MAIN APP ──────────────────────────────────────────────────────
st.title("Customer Churn Risk Predictor")
st.caption("Enter customer details to predict churn probability in real time.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account info")
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    payment  = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.subheader("Services")
    internet     = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_sec   = st.selectbox("Online security",  ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech support",     ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Billing")
    monthly_charges = st.slider("Monthly charges ($)", 18, 120, 65)
    total_charges   = monthly_charges * tenure
    st.metric("Estimated total charges", f"${total_charges:,.0f}")
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    senior    = st.selectbox("Senior citizen",    ["No", "Yes"])

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map  = {
    "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
    "Electronic check": 2,          "Mailed check": 3
}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
binary_map   = {"No": 0, "Yes": 1, "No internet service": 2}

input_data = {
    "gender": 0, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 0, "Dependents": 0,
    "tenure": tenure,
    "PhoneService": 1, "MultipleLines": 0,
    "InternetService": internet_map[internet],
    "OnlineSecurity":   binary_map[online_sec],
    "OnlineBackup": 0,  "DeviceProtection": 0,
    "TechSupport":      binary_map[tech_support],
    "StreamingTV":      binary_map[streaming_tv],
    "StreamingMovies": 0,
    "Contract":         contract_map[contract],
    "PaperlessBilling": binary_map[paperless],
    "PaymentMethod":    payment_map[payment],
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     float(total_charges)
}

input_df = pd.DataFrame([input_data])

st.divider()

if st.button("Predict churn risk", type="primary"):
    prob     = model.predict_proba(input_df)[0][1]
    risk_pct = round(float(prob) * 100, 1)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        if prob >= 0.7:
            st.error(f"HIGH RISK — {risk_pct}% churn probability")
        elif prob >= 0.4:
            st.warning(f"MEDIUM RISK — {risk_pct}% churn probability")
        else:
            st.success(f"LOW RISK — {risk_pct}% churn probability")

        st.progress(float(prob))
        st.divider()
        st.subheader("Recommended action")
        if prob >= 0.7:
            if contract == "Month-to-month":
                st.error("🚨 Offer annual contract at 20% discount immediately")
            elif internet == "Fiber optic":
                st.error("🚨 Escalate to senior support — service dissatisfaction likely")
            else:
                st.error("🚨 Assign dedicated customer success manager")
        elif prob >= 0.4:
            st.warning("⚠️ Schedule proactive check-in call within 7 days")
        else:
            st.success("✅ No action needed — monitor monthly")

    with col_b:
        st.subheader("Why is this customer at risk? (SHAP)")
        st.caption("Red = increases churn risk  |  Blue = reduces churn risk")

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        feature_names = artifact["features"]
        shap_vals     = shap_values[0]

        indices      = np.argsort(np.abs(shap_vals))[-8:]
        sorted_vals  = shap_vals[indices]
        sorted_names = [feature_names[i] for i in indices]
        colors       = ["#C0392B" if v > 0 else "#2980B9" for v in sorted_vals]

        ax.barh(sorted_names, sorted_vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on churn probability)")
        ax.set_title("Feature contributions for this customer")
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        top_risk    = [(feature_names[i], shap_vals[i])
                       for i in np.argsort(shap_vals)[-3:][::-1] if shap_vals[i] > 0]
        top_protect = [(feature_names[i], shap_vals[i])
                       for i in np.argsort(shap_vals)[:2]         if shap_vals[i] < 0]

        if top_risk:
            st.caption("**Main risk factors:** " +
                       ", ".join([f"{n} (+{v:.3f})" for n, v in top_risk]))
        if top_protect:
            st.caption("**Protective factors:** " +
                       ", ".join([f"{n} ({v:.3f})" for n, v in top_protect]))