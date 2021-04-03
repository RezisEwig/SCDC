import pandas as pd
from library import *
import time

train_x_name = "train_preprocess.csv"
val_x_name = "val_preprocess.csv"
train_y_name = "en_train_y.csv"
val_y_name = "en_val_y.csv"


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


for i in range(10):
    start = time.time()
    X_sub, y_sub = get_minority_samples(x_train, y_train)  # Getting minority samples of that datframe
    X_res, y_res = MLSMOTE(X_sub, y_sub, 100, 5)  # Applying MLSMOTE to augment the dataframe
    x_train = pd.concat([x_train, X_res], axis=0)
    y_train = pd.concat([y_train, y_res], axis=0)
    end = time.time() - start
    print("cycle =",i+1)
    print("걸린 시간 : ", int(end/60),"min  ", int(end%60) ,"sec")


x_train.to_csv('MLSMOTE_x_train.csv', sep=',')
y_train.to_csv('MLSMOTE_y_train.csv', sep=',')

