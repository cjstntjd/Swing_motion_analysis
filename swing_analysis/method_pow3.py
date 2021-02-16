import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import axes3d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import sklearn.naive_bayes as nb
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import LSTM,Conv1D,MaxPooling1D
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow.keras.backend as K 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout,Input,Dense,Activation,Flatten,SeparableConv2D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import TimeDistributed

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)

class load:
    def __init__(self):
        self.file_name_dir = []
        self.total_data = []
        self.total_label = []

    def load_file(self,dir_location):
        print('now loading_file (location : ' + dir_location + ') ... \n')

        for root,dirs,files in os.walk(dir_location):
            for fname in files:
                full_fname = os.path.join(root,fname)
                self.file_name_dir.append(full_fname)

        print('make file list complete')

    def make_DataFrame(self,tar_li):
        for file_name in tqdm(self.file_name_dir):
            sp = file_name.split('/')
            tmp_label = sp[1]
            d = open(file_name,'r',encoding='UTF8').read()
            data = d.split('\n')
            data.pop(0) # remove trash data header
            index = data.pop(0)
            tmp_real_data = []
            for dat_num in range(len(data)):
                if data[dat_num] == '':
                    continue
                tmp_real_data.append(data[dat_num].split(','))

            df = pd.DataFrame(tmp_real_data)
            index_li = index.split(',')
            df.columns = index_li

            #now change str to float

            for y in index_li:
                df[y] = pd.to_numeric(df[y],downcast='float')

            tmp_li = []
            for i in range(len(df)):
                tmp = []
                for j in tar_li:
                    tmp.append((df[j][i]/1000)**3)
                tmp_li.append(tmp)

            self.total_data.append(tmp_li)
            self.total_label.append(tmp_label)
        print('make total_data finish.....')

    def return_data(self):
        return self.total_data , self.total_label







class Train_model:
    
    def __init__(self):
        
        self.encoder = LabelEncoder()
        self.enc_label = 0
        
        self.total_data = 0
        self.total_label = 0
        
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        
        self.earlystopping = EarlyStopping(monitor='val_loss',patience=10)
        
        #model list
        self.lstm = 0
        self.svm = 0
        self.xgboost = 0
        self.nb = 0
        self.rf =0
        self.knn = 0
        
        #sample prediction
        self.sample_data = 0
        self.sample_label = 0
        
    def get_enc(self):
        self.enc_label = self.encoder.fit_transform(self.total_label)
    
    def make_arr(self):
        self.total_data = np.array(self.total_data)
        self.enc_label =np.array(self.enc_label)
    
    def divide_dataset(self,mode):
        
        if mode == 'lstm':
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.total_data,self.enc_label,test_size=0.2,random_state=0)

        else:
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.total_data,self.enc_label,test_size=0.2,random_state=0)
            nsamples,nx,ny = self.x_train.shape
            self.x_train = self.x_train.reshape((nsamples,nx*ny))
            nsamples,nx,ny = self.x_test.shape
            self.x_test = self.x_test.reshape((nsamples,nx*ny))
        
    def model_create_train(self,mode):
        if mode == 'lstm':
            with tf.device('/GPU:0'):
                model = Sequential() # Sequeatial Model 
                model.add(LSTM(180, input_shape=(60,3),return_sequences = True)) # (timestep, feature) 
                model.add(Dropout(0.2))
                model.add(Conv1D(128,
                                 2,
                                 padding='valid',
                                 activation='relu',
                                 strides=1))
                model.add(MaxPooling1D(pool_size=4))
                model.add(LSTM(128))
                model.add(Dense(8, activation='softmax'))

                # 3. 모델 학습과정 설정하기
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                hist = model.fit(self.x_train, self.y_train, epochs=100, batch_size=256,callbacks=[self.earlystopping] ,validation_data=(self.x_test, self.y_test))
                model.save('model_x.h5')
                model.save_weights('model_x_weights.h5')
            self.lstm = model
            
        elif mode=='svm':
            #####here to change####
            ######################################################################

            best_score = 0

            for gamma in [0.001,0.01,0.1,1,10,100]:
                for C in [0.001,0.01,0.1,1,10,100]:
                    for kernel in ['linear','rbf','poly']:
                        tmp_model = svm.SVC(kernel=kernel,gamma=gamma,C=C)
                        scores = cross_val_score(tmp_model,self.x_train,self.y_train,cv=10,n_jobs=-1)
                        score = np.mean(scores)

                        if score>best_score:

                            best_score = score
                            best_parameter = {'kernel':kernel,'gamma':gamma,'C':C}
                            print('best_parameter is change : ',best_parameter)
                        else:
                            print('remain :',best_parameter)
            mod = svm.SVC(**best_parameter)
            

            ######################################################################
            
            predict_model = mod.fit(self.x_train,self.y_train)
            print('fitting ',mode,' is complete...')
            print(mode,'score is :',predict_model.score(self.x_test,self.y_test))

            prediction = predict_model.predict(self.x_test)
            self.svm = mod
        
        elif mode=='xgboost':
            #####here to change####
            ######################################################################

            space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
                    'gamma': hp.uniform ('gamma', 1,9),
                    'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
                    'reg_lambda' : hp.uniform('reg_lambda', 0,1),
                    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
                    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
                    'n_estimators': 180,
                    'seed': 0
                }

            mod =xgb.XGBClassifier(
                                n_estimators =space['n_estimators'], max_depth = space['max_depth'], gamma = space['gamma'],
                                reg_alpha = space['reg_alpha'],min_child_weight=space['min_child_weight'],
                                colsample_bytree=space['colsample_bytree'])

            

            ######################################################################
            
            predict_model = mod.fit(self.x_train,self.y_train)
            print('fitting ',mode,' is complete...')
            print(mode,'score is :',predict_model.score(self.x_test,self.y_test))

            prediction = predict_model.predict(self.x_test)
            self.xgboost = mod
        
        elif mode=='nb':
            #####here to change####
            ######################################################################

            mod = GaussianNB()

            ######################################################################
            
            predict_model = mod.fit(self.x_train,self.y_train)
            print('fitting ',mode,' is complete...')
            print(mode,'score is :',predict_model.score(self.x_test,self.y_test))

            prediction = predict_model.predict(self.x_test)
            self.nb = mod
        
        
        elif mode=='rf':
            #####here to change####
            ######################################################################
            
            rfc=RandomForestClassifier(random_state=42)
            param_grid = { 
                'n_estimators': [10,15,20,30,40,50,100,200,300,500],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [3,4,5,6,7,8,9,10,11,12],
                'criterion' :['gini', 'entropy']
            }

            mod = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10,n_jobs=-1)

            
            ######################################################################
            
            predict_model = mod.fit(self.x_train,self.y_train)
            print('fitting ',mode,' is complete...')
            print(mode,'score is :',predict_model.score(self.x_test,self.y_test))

            prediction = predict_model.predict(self.x_test)
            self.rf = mod
        
        elif mode=='knn':
            #####here to change####
            ######################################################################
            
            leaf_size = list(range(1,50))
            n_neighbors = list(range(1,8))
            p=[1,2]

            hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

            knn = KNeighborsClassifier()

            mod = GridSearchCV(knn, hyperparameters, cv=10, n_jobs=-1)

            
            ######################################################################
            
            predict_model = mod.fit(self.x_train,self.y_train)
            print('fitting ',mode,' is complete...')
            print(mode,'score is :',predict_model.score(self.x_test,self.y_test))

            prediction = predict_model.predict(self.x_test)
            self.knn = mod
        
    
    def prediction(self,input_axis,mode):
        p = load()
        p.load_file('test_data')
        p.make_DataFrame(input_axis)
        self.sample_data , self.sample_label = p.return_data()
        
        if mode == 'lstm':
            self.sample_data = np.array(self.sample_data)
            sample_pred = self.lstm.predict(self.sample_data)
            sample_pred = np.argmax(sample_pred,axis=-1)
            lab = self.encoder.inverse_transform(sample_pred)
            
            hit = 0
            miss = 0
            answer=[]
            print('testing new data result :\n[answer]  -->  [predict err]')
            for x in range(len(lab)):
                if lab[x] == self.sample_label[x]:
                    hit+=1
                    answer.append(lab[x])
                else:
                    miss+=1
                    print(total_label[x],' --> ' ,lab[x],'        err_index number : ',x)


            print('hit: ',hit,' miss : ',miss,'percent : ',(100*hit)/(hit+miss))
        
        else:
            model_list = ['svm','knn','rf','nb','xgboost']
            match_list = [self.svm , self.knn , self.rf , self.nb , self.xgboost]
            
            for x in range(len(model_list)):
                if model_list[x] == mode:
                    mod = match_list[x]
                    print(mode + 'model match complete.....')
                
            self.sample_data = np.array(self.sample_data)
            nsamples , nx , ny = self.sample_data.shape
            sample = self.sample_data.reshape((nsamples,nx*ny))
            sample_pred = mod.predict(sample)
            lab = self.encoder.inverse_transform(sample_pred)
            
            hit = 0
            miss = 0
            answer=[]
            print('testing new data result :\n[answer]  -->  [predict err]')
            for x in range(len(lab)):
                if lab[x] == self.sample_label[x]:
                    hit+=1
                    answer.append(lab[x])
                else:
                    miss+=1
                    print(total_label[x],' --> ' ,lab[x],'        err_index number : ',x)


            print('hit: ',hit,' miss : ',miss,'percent : ',(100*hit)/(hit+miss))
            




total_data = []
total_label = []
tar_dir = 'swing'
input_axis = ['AX','AY','AZ']

def IO(tar_dir,input_axis):
    total_dat = []
    total_lab = []
    v = load()
    v.load_file(tar_dir)
    v.make_DataFrame(input_axis)
    total_dat,total_lab = v.return_data()
    return total_dat , total_lab

total_data , total_label = IO(tar_dir,input_axis)

def pipline(total_data,total_label,input_axis,mode):
    t = Train_model()
    t.total_data = total_data
    t.total_label = total_label
    t.get_enc()
    t.make_arr()
    t.divide_dataset(mode)
    t.model_create_train(mode)
    t.prediction(input_axis,mode)

pipline(total_data,total_label,input_axis,'lstm')
