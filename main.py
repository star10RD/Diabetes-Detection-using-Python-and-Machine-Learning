import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import streamlit as st
st.set_page_config(layout="wide")
col1,col2,col3 = st.columns(3)
image = Image.open("logo.png")
with col1:
    st.write("")

with col2:
    st.image(image, use_column_width=True)

with col3:
    st.write("")

st.markdown("<link rel='icon' type='image/x-icon' href='favicon.ico'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Detecting Diabetes using Machine Learning and Python</h1>", unsafe_allow_html=True) 
st.sidebar.markdown("<h1 style='text-align: center;'>User Input</h1>", unsafe_allow_html=True)

col1,col2 = st.columns(2)
df = pd.read_csv("Datasets.csv") #   Data processing 
with col1:
    st.subheader("Dataset: ")
    st.dataframe(df)
with col2:
    st.subheader("Data Description:")
    st.write(df.describe())

st.subheader("Bar Chart: ")
chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


def get_user_input():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin", 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.78, 2.42, 0.375)
    age = st.sidebar.slider("Age", 21, 82, 45)

    user_data = {"pregnancies": pregnancies,
                 "glucose": glucose,
                 "blood_pressure": blood_pressure,
                 "skin_thickness": skin_thickness,
                 "insulin": insulin,
                 "BMI": BMI,
                 "diabetes_pedigree_function": diabetes_pedigree_function,
                 "age": age
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()

st.subheader("User Input:")
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
KNN = KNeighborsClassifier(13)
KNN.fit(X_train, Y_train)
RandomForestClassifier.fit(X_train, Y_train)


st.subheader("Model Test Accuracy Score: ")
st.write("Random Forest: " +str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + "%")
st.write("K Neighbors: " +str(accuracy_score(Y_test, KNN.predict(X_test)) * 100) + "%")
rfc_pred = RandomForestClassifier.predict(user_input)
knn_pred = KNN.predict(user_input)
st.subheader("Classification: ")
col1,col2=st.columns(2)
with col1:
    st.write("Random Forest:")
    st.write(rfc_pred)

with col2:
    st.write("K Neighbors:")
    st.write(knn_pred)

st.markdown("<h2 style='text-align: center;'>Project Made By:</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Vedant Poddar - 2019-B-03032001</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Shrestha Sarda - 2019-B-12072000A</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Kastup Bhattarai - 2019-B-28092001B</h1>", unsafe_allow_html=True)