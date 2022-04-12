import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

def main():
  datatrain=pd.read_csv(r'data train numerik no box 20 k5 91.csv')

  datatrain["DERMAGA"]=datatrain["DERMAGA"].astype('category')
  datatrain["JENIS_KAPAL"]=datatrain["JENIS_KAPAL"].astype('category')
  datatrain["DELAY"]=datatrain["DELAY"].astype('category')
  datatrain["PALKA"]=datatrain["PALKA"].astype('int')
  datatrain["BD"]=datatrain["BD"].astype('int')
  datatrain["SHIFTING"]=datatrain["SHIFTING"].astype('int')
  datatrain["WAG"]=datatrain["WAG"].astype('int')
  datatrain["BAD_WEATHER"]=datatrain["BAD_WEATHER"].astype('int')
  datatrain["JUMLAH_CC"]=datatrain["JUMLAH_CC"].astype('int')
  datatrain["DISCHARGE"]=datatrain["DISCHARGE"].astype('int')
  datatrain["LOADING"]=datatrain["LOADING"].astype('int')
  
  arr = datatrain.values
  X_train = arr[:, 0:10]
  Y_train = arr[:, 10]

  datatest=pd.read_csv(r'data test numerik no box 20 k5 91.csv')

#tambahin ubah data integer
  datatest["DERMAGA"]=datatest["DERMAGA"].astype('category')
  datatest["JENIS_KAPAL"]=datatest["JENIS_KAPAL"].astype('category')
  datatest["DELAY"]=datatest["DELAY"].astype('category')
  datatest["PALKA"]=datatest["PALKA"].astype('int')
  datatest["BD"]=datatest["BD"].astype('int')
  datatest["SHIFTING"]=datatest["SHIFTING"].astype('int')
  datatest["WAG"]=datatest["WAG"].astype('int')
  datatest["BAD_WEATHER"]=datatest["BAD_WEATHER"].astype('int')
  datatest["JUMLAH_CC"]=datatest["JUMLAH_CC"].astype('int')
  datatest["DISCHARGE"]=datatest["DISCHARGE"].astype('int')
  datatest["LOADING"]=datatest["LOADING"].astype('int')

  arr = datatest.values
  X_test = arr[:, 0:10]
  Y_test = arr[:, 10]

  knn=KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train,Y_train)
  Y_PredKNN=knn.predict(X_test)

  filename = 'knn.sav'

  #Saving the Model
  pickle_out = open("knn.sav", "wb") 
  pickle.dump(knn, pickle_out) 
  pickle_out.close()

  pickle_in = open('knn.sav', 'rb')
  classifier = pickle.load(pickle_in)

  st.header('Prediksi Delay Keberangkatan Kapal')
  DERMAGA = st.selectbox("Dermaga Sandar:", ("1", "2", "3", "4"))
  JENIS_KAPAL = st.selectbox("Jenis Kapal:", ("Feeder", "Direct"))
  PALKA = st.number_input("Jumlah Palka (Unit):")
  JUMLAH_CC = st.number_input("Jumlah CC (Unit):")
  DISCHARGE = st.number_input("Jumlah Petikemas yang Dibongkar (Box):")
  LOADING = st.number_input("Jumlah Petikemas yang Dimuat (Box):")
  BD = st.number_input("Lama Waktu Breakdown (Menit):")
  SHIFTING_YARD = st.number_input("Jumlah Shifting Yard (Box):")
  WAG = st.number_input("Lama Waktu WAG (Menit):")
  BAD_WEATHER = st.number_input("Lama Waktu Bad Weather (Menit):")
  submit = st.button('Predict')
  if submit:
    if JENIS_KAPAL == "Direct":
      JENIS_KAPAL = 1
    else:
      JENIS_KAPAL = 0
    if DERMAGA == "1":
      DERMAGA = 1
    elif DERMAGA == "2":
      DERMAGA = 2
    elif DERMAGA == "3":
      DERMAGA = 3
    else:
      DERMAGA = 4
      
    prediction = classifier.predict([[DERMAGA, JENIS_KAPAL, PALKA, JUMLAH_CC, DISCHARGE, LOADING, BD, SHIFTING_YARD, WAG, BAD_WEATHER]])
    #tabel prediksi
    #st.write(prediction)
    if prediction == 0:
        st.write('KAPAL TIDAK MENGALAMI DELAY KEBERANGKATAN')
    elif prediction == 1:
        st.write('KAPAL MENGALAMI DELAY KURANG DARI 4 JAM')
    else:
        st.write('KAPAL MENGALAMI DELAY LEBIH DARI 4 JAM')

if __name__ == '__main__':
  main()
