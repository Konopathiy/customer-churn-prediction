import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "data/telco_churn.csv"

st.set_page_config(page_title="Прогноз оттока клиентов", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Обучаю модель на датасете...")
def train_model(path):
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.drop(columns=["customerID"], errors="ignore")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])

    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    lr.fit(X_train, y_train)

    metrics = {
        "rf_acc": accuracy_score(y_test, rf.predict(X_test)),
        "rf_auc": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
        "lr_acc": accuracy_score(y_test, lr.predict(X_test)),
        "lr_auc": roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]),
    }

    best = rf if metrics["rf_auc"] >= metrics["lr_auc"] else lr
    return best, list(X.columns), metrics


model, feature_columns, metrics = train_model(DATA_PATH)

st.title("📉 Прогноз оттока клиентов телеком-компании")
st.caption("Модель обучается при запуске приложения на датасете Telco Customer Churn.")

with st.sidebar:
    st.header("Метрики на тесте")
    st.metric("Random Forest — ROC-AUC", f"{metrics['rf_auc']:.3f}")
    st.metric("Logistic Regression — ROC-AUC", f"{metrics['lr_auc']:.3f}")
    st.caption(f"Точность RF: {metrics['rf_acc']:.3f} · LR: {metrics['lr_acc']:.3f}")

st.subheader("Данные клиента")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Пол", ["Female", "Male"])
    senior = st.selectbox("Пожилой клиент", ["Нет", "Да"])
    partner = st.selectbox("Есть партнёр", ["No", "Yes"])
    dependents = st.selectbox("Есть иждивенцы", ["No", "Yes"])
    tenure = st.slider("Срок сотрудничества (мес.)", 0, 72, 12)

with col2:
    phone = st.selectbox("Телефонная связь", ["Yes", "No"])
    multiple = st.selectbox("Несколько линий", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Тип интернета", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Онлайн-безопасность", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Онлайн-бэкап", ["No", "Yes", "No internet service"])

with col3:
    protection = st.selectbox("Защита устройства", ["No", "Yes", "No internet service"])
    support = st.selectbox("Техподдержка", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Стриминг ТВ", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Стриминг фильмов", ["No", "Yes", "No internet service"])

col4, col5, col6 = st.columns(3)

with col4:
    contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Электронные счета", ["Yes", "No"])

with col5:
    payment = st.selectbox(
        "Способ оплаты",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Месячный платёж ($)", 0.0, 200.0, 70.0, step=1.0)

with col6:
    total = st.number_input(
        "Общая сумма ($)", 0.0, 10000.0, float(round(monthly * max(tenure, 1), 2)), step=10.0
    )

if st.button("Спрогнозировать", type="primary"):
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Да" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    X_new = pd.DataFrame([row])
    X_new = pd.get_dummies(X_new)
    X_new = X_new.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(X_new)[0, 1]
    pred = int(proba >= 0.5)

    st.divider()
    if pred == 1:
        st.error(f"Клиент, скорее всего, уйдёт. Вероятность оттока: {proba:.1%}")
    else:
        st.success(f"Клиент, скорее всего, останется. Вероятность оттока: {proba:.1%}")

    st.progress(min(max(proba, 0.0), 1.0))

    st.subheader("Что влияет на прогноз")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_columns)
        imp = imp.sort_values(ascending=False).head(10)
        st.bar_chart(imp)
    else:
        st.info("Для логистической регрессии важность признаков не отображается.")
