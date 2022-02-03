import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Any, Dict, List
import load_config




    
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


def cross_val(model):
    pred = cross_val_score(model, x, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


########### Sidebar ###########
# 1. Upload Dataset
st.sidebar.title('1. Loading Data')
if st.sidebar.checkbox("Load Existing Dataset", value = True) == True: error_message = """
Your CSV file should gained from the G Net Track app and have at least the following column:
* __Timestamp__: gagal upload timestamp
"""
  uploaded_files = st.sidebar.selectbox('Choose a CSV files',  
                                       ('S9-9am-20191124.csv', 
                                        'S9-12pm-20191124.csv', 
                                        'S9-6pm-20191124.csv', 
                                        'S10e-9am-20191124.csv', 
                                        'S10e-12pm-20191124.csv', 
                                        'S10e-6pm-20191124.csv'))

else : 
  uploaded_files = st.sidebar.file_uploader("Choose a file", type={"csv","txt"})
  agree = False
  agree2 = False 
  pca_comp = False

#membuat dataframe kemudian nantinya digunakan untuk proses selanjutnya pada data preparation
if uploaded_files and uploaded_files!='None' :
  if type.uploaded_files == "csv":
    df = df.fillna(0)
    df = df[['Timestamp', 'Longitude','Latitude','Speed','Operator','CellID','LAC','LTERSSI','RSRP','RSRQ','SNR','DL_bitrate','UL_bitrate']]
    df_selected = True
  else :
    uploaded_files.to_csv("{}".format("".txt))
    df = df.fillna(0)
    df = df[['Timestamp', 'Longitude','Latitude','Speed','Operator','CellID','LAC','LTERSSI','RSRP','RSRQ','SNR','DL_bitrate','UL_bitrate']]
    df_selected = True

  
 #opsional dan masih ambigu untuk fitur ini sebenarnya untuk membuat target column seperti apa, tetapi dibuat terlebih dahulu untuk kalau misalkan diperlukan dalam launch forecast
  date_time = st.sidebar.selectbox('Date Time', df.columns[0:1])
  target_time = st.sidebar.selectbox("Target Columns", df.columns[1:])

#2. Data Preparation
#dalam melakukan preprocessing, dataframe sebelumnya akan dilakukan korelasi antar parameter untuk menentukan korelasi yang paling lemah dan tidak menggunakan correlation heatmap matriks
  st.sidebar.title('2. Data Preparation')       #selected_filename = st.sidebar.selectbox('Select a file', [df]) #selected_filename = st.sidebar.selectbox('Select a file', ["CSV_1", "CSV_2", "CSV_3", "CSV_4", "CSV_5", "CSV_6",])
  agree = st.sidebar.checkbox("Launch Heatmap", value=False)
  with st.sidebar.expander("Data Dimension", expanded=True): 
     features = st.sidebar.multiselect("Choose a Feature", df.columns[0:]) #setelah mempelajari hasil korelasi heatmap, kemudian akan dipilih fitur apa yang memiliki korelasi yang kuat untuk diproses ke dalam preprocessing selanjutnya yaitu pada PCA
     excluded = st.sidebar.multiselect('Select Excluded Features', features) #feature excludes ini nantinya digunakan ketika kita ingin menngexclude fitur pada PCA 

  agree2 = st.sidebar.checkbox("Launch PCA")
  pca_comp = st.sidebar.number_input('Select Number of PC', min_value=1, max_value=len(features)-len(excluded), step=1)  

  if(pca_comp):
    agree3 = st.sidebar.checkbox('Transform?')  #fitur transform dibuat agar kita dapat memastikan apakah fitur exclude nanti dapat dipertimbangkan untuk digunakan kedalam PCA atau tidak
  
  if(agree3):
    split = st.sidebar.slider('Training-Test Split (%)', 1, 99, 50)*0.01 #setelah melalui PCA, data akan dibuat untuk modelling, data data tersebut yaitu data training dan data test
    Table = df.set_index('Timestamp')
    training_data, testing_data = train_test_split(Table, test_size=1-split)  
    st.sidebar.write("Number of training examples = " + str(training_data.shape[0]))
    st.sidebar.write("Number of testing examples = " + str(testing_data.shape[0]))


  st.sidebar.title('3. Modelling') 
  regresi = st.sidebar.selectbox('Choose Regression', ('None', 'kNN','Random Forest','AdaBoost'))
  if regresi == 'kNN':
    k_param = st.sidebar.slider('k (slider)', 0, 10)
    metrics = st.sidebar.selectbox('Metrics', ('minkowski', 'euclidean', 'manhattan', 'chebysev'))
  elif regresi == 'Random Forest':
    n_estimators = st.sidebar.slider('n estimator', 0, 10)
    criterion = st.sidebar.selectbox('Criterion', ('squared_error', 'absolute_error', 'poisson'))
    max_depth = st.sidebar.number_input('max depth', min_value=1, max_value=3, step=2)  
    min_samples_split = st.sidebar.number_input('min samples split', min_value=1, max_value=3, step=2)  
  elif regresi == 'AdaBoost':
    n_estimators = st.sidebar.slider('n estimator', 0, 10)
    learning_rate = st.sidebar.slider('learning rate', 0, 10)
    loss = st.sidebar.selectbox('loss', ('linear', 'square', 'exponential'))
  # elif regresi == 'Gradient Boost':
  #   st.sidebar.slider('n estimator', 0, 10)
  #   st.sidebar.slider('learning rate', 0, 10)
  #   st.sidebar.selectbox('loss', ('linear', 'square', 'exponential'))
  #   st.sidebar.number_input('depth', min_value=1, max_value=3, step=2)
  # elif regresi == 'SVM':
  #   st.sidebar.number_input('epsilon', min_value=1, max_value=3, step=2)
  #   st.sidebar.slider('C', 0, 10)


st.markdown("""
                    <h1 style='text-align: center;'>\
                        Main Dashboard 4G LTE</h1>
                    """, 
                    unsafe_allow_html=True)
st.write("")

with st.expander("Apa itu Main Dashboard 4G LTE?", expanded=False): #fitur berupa penjelasan dashboard kepada pengguna yang ingin mempelajari proses prediksi throughput menggunakan machine learning
    st.markdown('Main Dashboard 4G LTE merupakan dashboard yang dirancang dengan tujuan untuk melakukan train data, cleaning data, membentuk data baru lalu kemudian dimodelkan menggunakan beberapa tipe model machine learning untuk menampilkan hasil prediksi throughput 4G LTE dalam bentuk grafik linear dan chart, serta menunjukkan evaluasi model menggunakan metrics:')
    st.write("")
with st.expander("Bagaimana cara memulai menggunakan dashboard ini?", expanded=False): #step pengguna yang ingin menggunakan dashboard dalam memproses data throughput menggunakan machine learning
    st.markdown('Berikut adalah beberapa bagian utama dalam Main Dashboard 4G LTE :')
    st.write('* __Loading Data__: Dapat menggunakan dataset existing atau upload, menentukan target dan date column.')
    st.write('* __Data Preparation__: Dapat melakukan Data Preprocessing sebelum membentuk menjadi data yang siap digunakan untuk pemodelan machine learning.')
    st.write('* __Modelling__: Setelah melalui Data Preprocessing, nantinya data tersebut akan digunakan untuk pemodelan machine learning menggunakan beberapa tipe model machine learning.')
    st.write('* __Launch Visualization__: Menampilkan hasil prediksi throoughput untuk kondisi actual dan predicted menggunakan beberapa tipe model machine learning/ ')
    st.write("")
st.write("")  

st.checkbox('Launch Forecast')

if agree == True:
    st.markdown("""
                    <h1 style='text-align: center;'>\
                        Correlation Matriks untuk Fitur Numerik</h1>
                    """, 
                    unsafe_allow_html=True)
    st.write("")
    # Correlation Score
    fig = plt.figure(figsize=(10, 8))
    # st.write(df[['Longitude','Latitude','Speed','Operator','CellID','LAC','LTERSSI','RSRP','RSRQ','SNR','DL_bitrate','UL_bitrate']].corr())
    correlation_matrix = df.corr().round(2)
    # annot = True to print the values inside the square
    sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
    st.write(fig)
    if(features!=[]):
       df = df[['Timestamp'] + features]
       st.write(df)

st.markdown("")

if agree and agree2:
  if(len(excluded) != 0):
    features = [x for x in features if x not in excluded]
  # EXPLORATORY DATA ANALYSIS


  # fig2 = sns.pairplot(df[features], plot_kws={"s": 3});
  # st.markdown("""
  #                   <h1 style='text-align: center;'>\
  #                       Exploratory Data Analysis</h1>
  #                   """, 
  #                   unsafe_allow_html=True)
  # st.write("")
  # st.pyplot(fig2)
  pca = PCA(n_components=pca_comp, random_state=123)
  pca.fit(df[features])
  princ_comp = pca.transform(df[features])
  st.write(princ_comp)
  st.write(pca.explained_variance_ratio_.round(3))

  if(agree3):
    pca = PCA(n_components=1, random_state=123)
    pca.fit(df[features])
    df['performance'] = pca.transform(df.loc[:, (features)]).flatten()
    df.drop(features, axis=1, inplace=True)
    st.write(df)


    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors="coerce")
    df['day'] =df['Timestamp'].dt.day
    df['month'] =df['Timestamp'].dt.month
    df['year'] =df['Timestamp'].dt.year
    df['hour'] =df['Timestamp'].dt.hour
    df['minute'] =df['Timestamp'].dt.minute
    df['second'] =df['Timestamp'].dt.second
    
    df = df[['Timestamp', 'DL_bitrate', 'performance', 'hour','minute','second']]
    df = df.fillna(0)
    Table = df.set_index('Timestamp')

    y = df['DL_bitrate']
    x = df.drop(columns=["DL_bitrate"])
    training_data, testing_data = train_test_split(Table, test_size=1-split)
    x_train, y_train = training_data.drop("DL_bitrate", axis=1), training_data['DL_bitrate']
    x_test, y_test   = testing_data.drop("DL_bitrate", axis=1) , testing_data['DL_bitrate']

    numerical_features = ['performance']
    scaler = StandardScaler()
    scaler.fit(x_train[numerical_features])
    x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
    x_train[numerical_features].head()

    models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])

    if regresi is not None:
      x_train, y_train = training_data.drop("DL_bitrate", axis=1), training_data['DL_bitrate']
      x_test, y_test   = testing_data.drop("DL_bitrate", axis=1) , testing_data['DL_bitrate']
      models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['kNN', 'RandomForest', 'Boosting'])
      prediksi = x_test.iloc[:30].copy()

      knn = RF = boosting = None

      if(regresi == 'kNN'):
        knn = KNeighborsRegressor(n_neighbors=k_param, metric = metrics)
        knn.fit(x_train, y_train)
        y_pred_knn = knn.predict(x_train)

        models.loc['train_mse','kNN'] = mean_squared_error(y_pred=knn.predict(x_train), y_true=y_train)

        pred_dict = {'y_true':y_test[:30]}
        pred_dict['prediksi_knn'] = knn.predict(prediksi).round(1)
        # #Evaluation
        # chart_data = pd.DataFrame(np.random.randn(10, 2), columns =['Actual', 'Prediction'])
        # arr = np.random.normal(1, 1, size=10)
        # fig4, y_pred_knn = plt.subplots()
        # y_pred_knn.hist(arr, bins=5)
      elif(regresi == 'Random Forest'):
        RF = RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth, criterion= criterion, min_samples_split = min_samples_split, random_state = 55, n_jobs = 1)
        RF.fit(x_train, y_train)
        y_pred_rf = RF.predict(x_train)

        models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)

        pred_dict = {'y_true':y_test[:30]}
        pred_dict['prediksi_rf'] = RF.predict(prediksi).round(1)
        
      elif(regresi == 'AdaBoost'):
        boosting = AdaBoostRegressor(n_estimators=n_estimators, learning_rate = learning_rate, random_state = 55, loss = loss)
        boosting.fit(x_train, y_train)
        y_pred_boosting = boosting.predict(x_train)
        
        models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(x_train), y_true=y_train)

        pred_dict = {'y_true':y_test[:30]}
        pred_dict['prediksi_boosting'] = boosting.predict(prediksi).round(1)

      st.write(models)
      x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])
      mse = pd.DataFrame(columns=['train', 'test'], index=['kNN','RF','Boosting'])
      if(knn!=None and RF!=None and boosting!=None):
        model_dict = {'kNN': knn, 'RF': RF, 'Boosting': boosting}
        for name, model in model_dict.items():
            mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3 
            mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3 

      # st.line_chart(chart_data)
      # st.pyplot(fig4)

 
 #testing with test data
       
      hasil = pd.DataFrame(pred_dict) 
      hasil=hasil.sort_values(by='Timestamp')

  
st.markdown("")
visual = st.selectbox('Choose Visualization', ('kNN','Random Forest','AdaBoost'))
if 'kNN' in visual:
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.set_xlabel('Time')
  ax.set_ylabel('Throughput')
  ax.set_title('Throughput Prediction vs Actual')
  ax.grid(True)

  # Plotting on the first y-axis
  ax.plot(hasil['y_true'], color='tab:orange', label='Actual')
  ax.plot(hasil['prediksi_knn'], color='tab:cyan', label='Prediction')
  ax.legend(loc='upper right');
     
  st.line_chart(hasil)

elif 'Random Forest' in visual:
  chart_data = pd.DataFrame(
  np.random.randn(10, 2),
  columns=['Actual', 'Prediction'])

  arr = np.random.normal(1, 1, size=10)
  fig3, ay = plt.subplots()
  ay.hist(arr, bins=5)

     
  st.line_chart(chart_data)
  st.pyplot(fig3)

elif 'AdaBoost' in visual:
  chart_data = pd.DataFrame(
  np.random.randn(10, 2),
  columns=['Actual', 'Prediction'])

  arr = np.random.normal(1, 1, size=10)
  fig3, ay = plt.subplots()
  ay.hist(arr, bins=5)

     
  st.line_chart(chart_data)
  st.pyplot(fig3)

elif 'Gradient Boost' in visual:
  chart_data = pd.DataFrame(
  np.random.randn(10, 2),
  columns=['Actual', 'Prediction'])

  arr = np.random.normal(1, 1, size=10)
  fig3, ay = plt.subplots()
  ay.hist(arr, bins=5)

     
  st.line_chart(chart_data)
  st.pyplot(fig3)

else:
  chart_data = pd.DataFrame(
  np.random.randn(10, 2),
  columns=['Actual', 'Prediction'])

  arr = np.random.normal(1, 1, size=10)
  fig3, ay = plt.subplots()
  ay.hist(arr, bins=5)

     
  st.line_chart(chart_data)
  st.pyplot(fig3)

mae = mean_absolute_error(y_true=y_test, y_pred=knn.predict(x_test))/1e3
mse = mean_squared_error(y_true=y_test, y_pred=knn.predict(x_test))/1e3
rmse = np.sqrt(mean_squared_error (y_true=y_test, y_pred=knn.predict(x_test)))/1e3
r2_square = r2_score(y_true=y_test, y_pred=knn.predict(x_test))
if st.checkbox("Launch Performance Metrics"):
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("MAE", mae)
  col2.metric("MSE", mse)
  col3.metric("RMSE",  rmse)
  col4.metric("R2 Score",  r2_square)