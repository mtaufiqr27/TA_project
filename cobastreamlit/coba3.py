import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


    
st.set_page_config(
  page_title="Dashboard Prediction 4G LTE",
  page_icon="🧊",
  layout="wide",
  initial_sidebar_state="expanded",
  menu_items={
     'Get Help': 'https://www.extremelycoolapp.com/help',
     'Report a bug': "https://www.extremelycoolapp.com/bug",
     'About': "# This is a header. This is an *extremely* cool app!"
    }

)


########### Sidebar ###########
    
st.sidebar.title('1. Data')
if st.sidebar.checkbox("Load Existing Dataset") == False:
# Upload Dataset
#with st.sidebar.expander("Upload Dataset", expanded=False):
    uploaded_files = st.sidebar.file_uploader("Choose a CSV file",  type="csv")
else : 
     # df = pd.read_csv(uploaded_files, encoding="utf-8")
     # df = df.fillna(0)
     # df = df.astype(str)
     # st.write(df)

    uploaded_files = st.sidebar.selectbox('Choose a CSV files',  ('None', 'S9-9am-20191124.csv', 'S9-12pm-20191124.csv', 'S9-6pm-20191124.csv', 'S10e-9am-20191124.csv', 'S10e-12pm-20191124.csv', 'S10e-6pm-20191124.csv'))

if uploaded_files and uploaded_files!='None' :         #selected_filename = st.sidebar.selectbox('Select a file', [df]) #selected_filename = st.sidebar.selectbox('Select a file', ["CSV_1", "CSV_2", "CSV_3", "CSV_4", "CSV_5", "CSV_6",])
  df = pd.read_csv(uploaded_files, encoding="utf-7")
  df = df.fillna(0)
  df = df.astype(str)
  st.write(df)


#else:
   # selected_filename = st.sidebar.selectbox('Select a file', ["CSV_1", "CSV_2", "CSV_3", "CSV_4", "CSV_5", "CSV_6",])

    #for uploaded_file in uploaded_files:
         #bytes_data = uploaded_file.read()
         #st.write("filename:", uploaded_file.name)
         #st.write(bytes_data)

    
st.sidebar.title('2. Modelling')
# Feature Selection
with st.sidebar.expander("Feature Selection", expanded=False):
  st.write(df)
  agree = st.sidebar.checkbox("Launch Heatmap Correlation", value=False)
  
  

# PCA
with st.sidebar.expander("PCA", expanded=False):
	y = st.slider('Choose Number PCA', min_value=0.0, max_value=6.0, step=1.0)

st.sidebar.title('3. Predictor')
if st.sidebar.button('Select Predictor Manually'):
    st.sidebar.selectbox("Choose Predictor", ["KNN", "AdaBoost", "Random Forest"])
    with st.sidebar.expander("Metrics", expanded=False):
        metric = ("RMSE", "MSE", "MAE", "R2")
        option_metric = st.multiselect("Choose Metrics",metric, default="RMSE")

st.markdown("""
                    <h1 style='text-align: center;'>\
                        Main Dashboard 4G LTE</h1>
                    """, 
                    unsafe_allow_html=True)
st.write("")

with st.expander("What is this app?", expanded=False):
    st.markdown('This app allows you to train, evaluate and optimize a Prophet model in just a few clicks. All you have to do is to upload a time series dataset, and follow the guidelines in the sidebar to :')
    st.write("")
with st.expander("How to use this app?", expanded=False):
    st.markdown('Berikut adalah langkah atau cara menggunakan dashboard ini :')
    st.write('* __Prepare data__: Filter, aggregate, resample and/or clean your dataset.')
    st.write('* __Choose model parameters__: Default parameters are available but you can tune them. Look at the tooltips to understand how each parameter is impacting forecasts.')
    st.write('* __Select evaluation method__: Define the evaluation process, the metrics and the granularity to assess your model performance.')
    st.write('* __Make a forecast__: Make a forecast on future dates that are not included in your dataset, with the model previously trained. ')
    st.write("")
st.write("")	

st.checkbox('Launch Forecast')


data_trip_9am_1 = pd.read_csv('S9-9am-20191124.csv', encoding = 'utf-7')
data_trip_9am_1 = data_trip_9am_1[['Longitude', 'Latitude', 'Speed', 'LTERSSI', 'RSRP', 'RSRQ', 'SNR','DL_bitrate','UL_bitrate']]
if agree == True:
    plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
    # Correlation Score
    fig = plt.figure(figsize=(10, 8))
    correlation_matrix = data_trip_9am_1.corr()
# annot = True to print the values inside the square
    sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
    st.write(fig)

 
#fig2 = sns.pairplot(data_trip_9am_1[['LTERSSI','RSRP','SNR']], plot_kws={"s": 3});
#st.selec  

