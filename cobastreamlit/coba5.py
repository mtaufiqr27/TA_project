  # # Correlation Score
    # plt.figure(figsize=(10, 8))
    # correlation_matrix = data_trip_9am_1.corr().round(2)
    # # annot = True to print the values inside the square
    # sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
    # plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)


# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot

# plotting correlation heatmap

  
# displaying heatmap

   # Correlation Score
  #   fig =  plt.subplots(figsize=(20,15))
  #   corr = df.corr()
  #   # st.write(df)
  #   sns.heatmap(corr,  annot=True, cmap='coolwarm', linewidths=0.5, )
  # # # Cannot = True to print the values inside the square
  #   # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
  #   st.pyplot(fig)
    

    #Select PC (input number) 2x

  # data_trip_9am_1 = pd.read_csv('S9-9am-20191124.csv', encoding = 'utf-7')
  # data_trip_9am_1 = data_trip_9am_1[['Longitude', 'Latitude', 'Speed', 'LTERSSI', 'RSRP', 'RSRQ', 'SNR','DL_bitrate','UL_bitrate']]
  data_trip = pd.read_csv(uploaded_files, encoding = 'utf-7')
  data_trip = data_trip[['Longitude', 'Latitude', 'Speed', 'LTERSSI', 'RSRP', 'RSRQ', 'SNR','DL_bitrate','UL_bitrate']]
  if agree == True:
      st.title("Correlation Matrix untuk Fitur Numerik")
    # Correlation Score
      fig = plt.figure(figsize=(10, 8))
      correlation_matrix = data_trip.corr()
# annot = True to print the values inside the square
      sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.008, )
      st.write(fig)



temp = ""
  for col in dimension_cols:
        temp = temp + "\'" + col + "\'" + ","
  temp = temp[:-1]
  temp = "df = df[[" + temp + "]]"
  exec(temp)
  st.write(df)
temp_5 = ""
  for col in dimension_cols:
      if col != date_col and col != target_col:
        temp_5 = temp_5 + "\'" + col + "\'" + ","
  temp_5 = temp_5[:-1]
  temp_5 = "numerical_features = [" + temp + "]"
  exec(temp_5)
  
  if perform_pca:
      numerical_features = ['performance']
      num_pca = st.number_input(
        'The minimum values is an integer of 3 or more.', value=3, step=1, min_value=2, help=readme['tooltips']['error_message'])
      jumlah_kolom = len(dimensions_cols)
      if jumlah_kolom < 5:
          st.warning('We need at least 3 features without the date and the target column!')
      else:
          pca = PCA(n_components=num_pca)
          scaler = StandardScaler()

##### I N F O R M A T I O N  D I S T R I B U T I O N (PC) #####

      temp_1 = ""

      for col in dimension_cols:
              if col != date_col and col != target_col:
                    temp_1 = temp_1 + "\'" + col + "\'" + ","

      temp_1 = temp_1[:-1]
      temp_1 = "pca.fit(df[[" + temp_1 + "]])"
      exec(temp_1)

      hasil_pca = pca.explained_variance_ratio_
      st.caption('PCs Information Proportion of the PCA')
      st.write(hasil_pca)   

############ D E T E R M I N E  P C A  D I M E N S I O N ############                     
  

      num_pca_2 = st.number_input('We need to set the number to 1 for reducing dimension to 1-dimension feature ',
        value=1, step=1, min_value=1, help=readme['tooltips']['error_message'])
      pca_2 = PCA(n_components=num_pca_2)

      temp_2 = ""
      for col in dimensions_cols:
              if col != date_col and col != target_col:
                      temp_2 = temp_2 + "\'" + col + "\'" + ","
      temp_2 = temp_2[:-1]
      temp_2 = "pca_2.fit(df[[" + temp_2 + "]])"
      exec(temp_2)

############ D E T E R M I N E  R E D U C T I O N  (P C A) ############ 

      temp_3 = ""

      for col in dimension_cols:
              if col != date_col and col != target_col:
                    temp_3 = temp_3 + "\'" + col + "\'" + ","

      temp_3 = temp_3[:-1]
      temp_3 = "df['performance'] = pca_2.transform(df.loc[:,(" + temp_3 + ")]).flatten()"
      exec(temp_3)

      temp_4 = ""

      for col in dimension_cols:
              if col != date_col and col != target_col:
                    temp_4 = temp_4 + "\'" + col + "\'" + ","

      temp_4 = temp_4[:-1]
      temp_4 = "df.drop([" + temp_4 + "], axis=1, inplace=True)"
      exec(temp_4)

      hasil_pca = pca.explained_variance_ratio_
      st.write(df)   

with st.sidebar.expander("Train-Test Split"):
        test_size = st.slider('% Size of the test split:', 1, 99, help=readme['tooltips']['train_test_split'])
        Table = df.set_index(date_col)
        training_data, testing_data = train_test_split(Table, test_size=0.01*test_size)
        x_train, y_train = training_data.drop(target_col, axis=1), training_data[target_col]
        x_test, y_test = testing_data.drop(target_col, axis=1), testing_data[target_col]

#execution time
        ct0 = datetime.now(tz=None)
        t0 = ct0.timestamp()
        knn = kNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
        knn.fit(x_train, y_train)
        ct1 = datetime.now(tz=None)
        t1=ct1.timestamp()
        duration = t1 - t0

        mae = mean_absolute_error(y_true=y_test, y_pred=knn.predict(x_test))/1e3
        mse = mean_squared_error(y_true=y_test, y_pred=knn.predict(x_test))/1e3
        rmse = np.sqrt(mean_squared_error (y_true=y_test, y_pred=knn.predict(x_test)))/1e3
        r2_square = r2_score(y_true=y_test, y_pred=knn.predict(x_test))

        st.sidebar.subheader('Forecast Horizon')
                  horizon = st.sidebar.slider('Select range to predict', 5, 50)
                  df_pred['y_true'] = y_test.iloc[:horizon]
                  df_pred['prediksi_KNN'] = knn.predict(x_test.iloc[:horizon].copy().round(1))
                  df_pred=df_pred.sort_values(by='Timestamp')
                  st.line_chart(df_pred)

        st.sidebar.subheader('Forecast Horizon')
                  horizon = st.sidebar.slider('true', 1, 3000, value=25, help="Tune the true vlue")
                  time = st.sidebar.slider('prediction', 1, 3000, value=25, help=" Tune the predicted value")

                  fig, ax = plt.subplot()
                  ax.set_xlabel('Time')
                  ax.set_ylabel('Throughput')
                  ax.set_title('Throughput Prediction vs Actual')
                  ax.grid(True)
                  df_pred['y_true'] = y_test
                  df_pred['prediksi_KNN'] = knn.predict(x_test.copy().round(1))
                  dfr_pred = df_pred.sort_values(by='Timestamp')
                  # Plotting on the first y-axis
                  ax.plot(df_pred[y_true].iloc[:time], color='tab:orange', label='Actual')
                  ax.plot(df_pred['prediksi_KNN'].iloc[:horizon], color='tab:orange', label='Prediction')
                  ax.legend(loc='upper right')
                  st.pyplot(fig)


