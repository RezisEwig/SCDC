import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model.model_gen import Deep_Model1, Deep_Model2


train_x_name = "preprocess/MLSMOTE_x_train.csv"
val_x_name = "preprocess/val_preprocess.csv"
train_y_name = "preprocess/MLSMOTE_y_train.csv"
val_y_name = "preprocess/en_val_y.csv"


raw_dataframe = pd.read_csv(train_x_name) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력
del raw_dataframe['cst_id_di'] # 첫째열와 행 제거
x_train = raw_dataframe


raw_dataframe = pd.read_csv(val_x_name) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력
del raw_dataframe['cst_id_di'] # 첫째열와 행 제거
x_valid = raw_dataframe


raw_dataframe = pd.read_csv(train_y_name) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력
del raw_dataframe['cst_id_di'] # 첫째열와 행 제거
y_train = raw_dataframe


raw_dataframe = pd.read_csv(val_y_name) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력
del raw_dataframe['cst_id_di'] # 첫째열와 행 제거
y_valid = raw_dataframe


rs=RobustScaler()
x_train2 = rs.fit_transform(x_train)
x_valid2 = rs.fit_transform(x_valid)


np_x_train=np.array(x_train)
np_y_train=np.array(y_train)
np_x_valid=np.array(x_valid)
np_y_valid=np.array(y_valid)

np_x_train2=np.array(x_train2)
np_x_valid2=np.array(x_valid2)


model = Deep_Model1()
model2 = Deep_Model2()


filename = 'model/best.h5'
rl=ReduceLROnPlateau(patience = 5, verbose=1, factor=0.5)
es=EarlyStopping(patience=20,verbose=1) # val_loss가 안좋아지면 멈춤, patience : val_loss값이 이전보다 감소하면 멈춤, 몇 번 감소하는지 설정
mc=ModelCheckpoint(filename,save_best_only=True,verbose=1) # verbose : 분석 과정에 결과 띄움


model.compile(optimizer=keras.optimizers.Adam(lr=0.005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model2.compile(optimizer=keras.optimizers.Adam(lr=0.005),
              loss='binary_crossentropy',
              metrics=['accuracy'])


mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for train_index, test_index in mskf.split(x_train,y_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_valid = np_x_train[train_index], np_x_train[test_index]
    Y_train, Y_valid = np_y_train[train_index], np_y_train[test_index]
    
    model.fit(X_train,Y_train,epochs=30,validation_data=(np_x_valid,np_y_valid),callbacks=[es,mc,rl])


filename = 'model/best2.h5'
rl=ReduceLROnPlateau(patience = 5, verbose=1, factor=0.5)
es=EarlyStopping(patience=20,verbose=1) # val_loss가 안좋아지면 멈춤, patience : val_loss값이 이전보다 감소하면 멈춤, 몇 번 감소하는지 설정
mc=ModelCheckpoint(filename,save_best_only=True,verbose=1) # verbose : 분석 과정에 결과 띄움


mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for train_index, test_index in mskf.split(x_train2,y_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_valid = np_x_train2[train_index], np_x_train2[test_index]
    Y_train, Y_valid = np_y_train[train_index], np_y_train[test_index]
    
    model.fit(X_train,Y_train,epochs=30,validation_data=(np_x_valid2,np_y_valid),callbacks=[es,mc,rl])