import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Hotel Booking Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/your-username/your-repo-name/main/hotel_bookings.csv")
    return df

df = load_data()

st.sidebar.title("Navigation")
tabs = ["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"]
choice = st.sidebar.radio("Go to", tabs)

# ---------- Data Visualization Tab ----------
if choice == "Data Visualization":
    st.header("Data Visualization Tab")
    st.write("Example visuals based on hotel booking data.")

    st.subheader("Hotel Type Distribution")
    st.bar_chart(df['hotel'].value_counts())

    st.subheader("Booking Cancellations")
    st.bar_chart(df['is_canceled'].value_counts())

    st.subheader("Average Daily Rate by Hotel")
    avg_adr = df.groupby("hotel")["adr"].mean()
    st.bar_chart(avg_adr)

    st.subheader("Top Countries by Booking Count")
    top_countries = df['country'].value_counts().head(10)
    st.bar_chart(top_countries)

# ---------- Classification Tab ----------
elif choice == "Classification":
    st.header("Classification Tab")

    df_class = df.copy()
    df_class = df_class.dropna(subset=['country'])

    features = ['lead_time', 'previous_cancellations', 'booking_changes',
                'is_repeated_guest', 'required_car_parking_spaces', 'total_of_special_requests']
    df_class = df_class[features + ['is_canceled']].dropna()

    X = df_class.drop('is_canceled', axis=1)
    y = df_class['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# ---------- Clustering Tab ----------
elif choice == "Clustering":
    st.header("Clustering Tab")
    st.write("To be implemented...")

# ---------- Association Rules Tab ----------
elif choice == "Association Rules":
    st.header("Association Rules Tab")
    st.write("To be implemented...")

# ---------- Regression Tab ----------
elif choice == "Regression":
    st.header("Regression Tab")
    st.write("To be implemented...")
