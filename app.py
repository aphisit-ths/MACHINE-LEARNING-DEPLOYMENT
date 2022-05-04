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
data = data.sample(frac=1)
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

    # -------------------------- ส่วนการ plot ต่างๆ
    if visuailize:
        for chart in visuailize:
            if chart == "Gender":
                st.write('### สัดส่วนเพศของข้อมูลขาเข้า :')
                gender_plot = go.Figure(data=[go.Pie(
                    labels=selection_row.gender.value_counts().index.tolist(),
                    values=selection_row.gender.value_counts().values.tolist())])
                st.plotly_chart(gender_plot, use_container_width=True)
            if chart == "Work Type":
                st.write('### สัดส่วนประเพศที่ทำงานของข้อมูลขาเข้า :')
                gender_plot = go.Figure(data=[go.Pie(
                    labels=selection_row.work_type.value_counts().index.tolist(),
                    values=selection_row.work_type.value_counts().values.tolist())])
                st.plotly_chart(gender_plot, use_container_width=True)
            if chart == "Smoking Status":
                st.write('### สัดส่วนสถานะการสูบบุหรี่ของข้อมูลขาเข้า :')
                gender_plot = go.Figure(data=[go.Pie(
                    labels=selection_row.smoking_status.value_counts().index.tolist(),
                    values=selection_row.smoking_status.value_counts().values.tolist())])
                st.plotly_chart(gender_plot, use_container_width=True)
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
    st.write('### การวัดผลการจัดกลุ่ม :')
    st_echarts(
        options=barChart, height="600px",
    )

# MODEL SECTION
