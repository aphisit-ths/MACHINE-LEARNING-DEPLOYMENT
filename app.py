import plotly.express as px
from streamlit_echarts import st_echarts
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
warnings.filterwarnings('ignore')


model_dtree = pickle.load(open('./treemodel.sav', 'rb'))
model_knn = pickle.load(open('./knn_model.sav', 'rb'))
model_rfr = pickle.load(open('./rfr_model.sav', 'rb'))


st.sidebar.header('Stroke Classification Web App')
st.title('Stroke Classification Web App')
st.write('>  Developed By : Sarayut Aree , Aphisit Thupsaeng')


def label_encoding(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])
    return df


data = pd.read_csv("./healthcare-dataset-stroke-data.csv")
data = data.sample(frac=1).reset_index(drop=True)
data.bmi.replace(to_replace=np.nan, value=data.bmi.mean(), inplace=True)
st.write('### Full Dataset', data)

# -------------------------- เลือก model ที่ใช้ทำนาย
st.sidebar.subheader("Choose Classifier")
model_selected = st.sidebar.selectbox(
    "Classifier", ("K-Nearest Neighbors (KNN)", "Random Forest (RF)", "Decision Tree"))
if model_selected == "Random Forest (RF)":
    model = model_rfr
elif model_selected == "Decision Tree":
    model = model_dtree
elif model_selected == "K-Nearest Neighbors (KNN)":
    model = model_knn


# -------------------------- เลือกว่าจะเอา input เข้าแบบไหน
st.sidebar.subheader("Choose input option for select data ")
input_option = st.sidebar.radio(
    "input option", ("Range of data", "Rows Selecting"))
st.sidebar.subheader("Select rows ")
if input_option == "Rows Selecting":
    selected_indices = st.sidebar.multiselect('Select rows:', data.index)
    selection_row = data.loc[selected_indices]
    input = selection_row.loc[selected_indices].iloc[:, 1:-1]
    target = selection_row.loc[selected_indices].iloc[:, -1]
else:
    option_values = st.sidebar.slider(
        'Select a range of values',
        0, len(data), (460, 785))
    selection_row = data.iloc[option_values[0]:option_values[1]]
    input = selection_row.iloc[:, 1:-1]
    target = selection_row.iloc[:, -1]

# -------------------------- เลือกว่าจะ plot อะไรบ้าง
visuailize = st.sidebar.multiselect(
    "What insight to plot?", ('Gender', "Work Type", "Smoking Status"))

if len(input) > 0:
    encode_df = label_encoding(input)
    prediction = model.predict(encode_df)
    accuracy = accuracy_score(target, prediction).round(2) * 100
    precision = precision_score(target, prediction).round(2) * 100
    recall = recall_score(target, prediction).round(2) * 100
    st.write('### Selecting Row : ', len(selection_row), selection_row)

    # -------------------------- ส่วนการ plot การวัดผล
    barChart = {
        "xAxis": {
            "type": "category",
            "data": ["accuracy",
                     "precision",
                     "recall"],
        },
        "yAxis": {"type": "value"},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
        },
        "series": [{"data": [accuracy, precision, recall], "type": "bar"}],
    }
    st.write('### วัดผลการจัดกลุ่ม :')
    st.write(" ##### Accuracy:  ", accuracy)
    st.write(" ##### Precision:  ", precision)
    st.write(" ##### Recall: ", recall)

    # With your data
    cm = confusion_matrix(target, prediction)
    st.write('### confusion_matrix :')
    # Visualizing Confusion Matrix
    fig = plt.figure(figsize=(8, 5))
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', linewidths=5, cbar=False, annot_kws={'fontsize': 15},
                yticklabels=['No stroke', 'Stroke'], xticklabels=['Predicted no stroke', 'Predicted stroke'])
    st.pyplot(fig)

   # -------------------------- ส่วนการ plot ต่างๆ
    if visuailize:
        for chart in visuailize:
            if chart == "Gender":
                st.write('### สัดส่วนเพศ :')
                gender_plot = go.Figure(data=[go.Pie(
                    labels=selection_row.gender.value_counts().index.tolist(),
                    values=selection_row.gender.value_counts().values.tolist())])
                st.plotly_chart(gender_plot, use_container_width=True)
            if chart == "Work Type":
                st.write('### สัดส่วนประเภทที่ทำงาน :')
                gender_plot = go.Figure(data=[go.Pie(
                    labels=selection_row.work_type.value_counts().index.tolist(),
                    values=selection_row.work_type.value_counts().values.tolist())])
                st.plotly_chart(gender_plot, use_container_width=True)
            if chart == "Smoking Status":
                st.write('### สัดส่วนสถานะการสูบบุหรี่ :')
                gender_plot = go.Figure(data=[go.Pie(
                    labels=selection_row.smoking_status.value_counts().index.tolist(),
                    values=selection_row.smoking_status.value_counts().values.tolist())])
                st.plotly_chart(gender_plot, use_container_width=True)

# ---------- test your data
st.header("Do you wanna try with you data ?")
own_data = st.checkbox('Yes , I do.')


def prepare_info():
    gender = st.radio("What's your gender ?",
                      ('Male', 'Female', 'Other'))
    age = st.slider("age", 16, 100, 16)
    hypertension = st.radio("hypertension ", (0, 1))
    heart_disease = st.radio("heart_disease ", (0, 1))
    ever_married = st.radio("ever_married ", ("No", "Yes"))
    work_type = st.radio(
        "work_type ", ("Private", "Self-employed ", "children", "Never_worked"))
    Residence_type = st.radio("Residence_type ", ("Urban", "Rural"))
    avg_glucose_level = st.slider("avg_glucose_level ", 50, 250, 50)
    bmi = st.slider("bmi ", 15, 40, 15)
    smoking_status = st.radio(
        "smoking_status ", ("never smoked", "Unknown", "formerly smoked", "smokes"))

    info = {
        "id": 99999,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    init_info = pd.DataFrame(info, index=[0])
    testdata = data.append(init_info, ignore_index=True)
    testdata_encode = label_encoding(testdata)
    return testdata_encode.iloc[-1:, 1:-1]


if own_data:
    userData = prepare_info()
    st.write('### Input Data', userData)
    test_prediction = model.predict(userData)
    isStroke = ""
    if test_prediction[0] == 1:
        isStroke = "มีความเสี่ยงเป็น Stroke"
    else:
        isStroke = "ไม่พบความเสี่ยง"
    st.write("## ผลการจัดกลุ่ม : ", isStroke)
