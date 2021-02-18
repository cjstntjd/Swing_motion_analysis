from concurrent import futures
import logging

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import grpc
import tensorflow as tf
import keras
import helloworld_pb2
import helloworld_pb2_grpc
from tensorflow.python.keras.backend import set_session
from sklearn.externals import joblib

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
set_session(session)
global lstm
global graph
graph = tf.get_default_graph()
lstm = load_model('/home/younghwan/swing_analysis/model_x.h5')
lstm.load_weights('/home/younghwan/swing_analysis/model_x_weights.h5')
lstm._make_predict_function()
global power
global mode
po_4 = ['lstm','rf','xgboost']
po_1 = ['nb','svm']
po_2 = ['knn']

global svm
global rf
global nb
global xgboost
global knn
svm = joblib.load('/home/younghwan/swing_analysis/svm_model.pkl')
rf = joblib.load('/home/younghwan/swing_analysis/rf_model.pkl')
nb = joblib.load('/home/younghwan/swing_analysis/nb_model.pkl')
xgboost = joblib.load('/home/younghwan/swing_analysis/xgboost_model.pkl')
knn = joblib.load('/home/younghwan/swing_analysis/knn_model.pkl')

global nu
nu = 0

class MLpredict(helloworld_pb2_grpc.MLpredictServicer):

    def Predict(self,request,context):
        global nu
        def analysis(d):
            tar_li = ['AX','AY','AZ']
            ind = ['back_cut', 'back_drive', 'back_short', 'back_smash', 'fo_cut' ,'fo_drive',
    'fo_short' ,'fo_smash']
            print(d)
            tmp_str = d.split('|')
            mode = tmp_str[0]
            if mode in po_4:
                power = 4
            elif mode in po_2:
                power = 2
            elif mode in po_1:
                power = 1

            data = tmp_str[1].split('\n')
            #data.pop(0)
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
                #tmp.append((df['AX'][i]/1000)**power)
                #tmp.append((df['AY'][i]/1000)**power)
                #tmp.append((df['AZ'][i]/1000)**power)
                tmp.append((df['AX'][i]))
                tmp.append((df['AY'][i]))
                tmp.append((df['AZ'][i]))
                tmp_li.append(tmp)
            total_data = []
            total_data.append(tmp_li)
            total_data = np.array(total_data)
            
            if mode != 'lstm':
                nsamples,nx,ny = total_data.shape
                total_data = total_data.reshape((nsamples,nx*ny))

            if mode == 'lstm':
                with graph.as_default():
                    set_session(session)
                    res = lstm.predict(total_data)
                res = np.argmax(res,axis=-1)
            elif mode == 'svm':
                res = svm.predict(total_data)
                print(res)
            
            elif mode == 'rf':
                res = rf.predict(total_data)
                print(res)
            
            elif mode == 'nb':
                res = nb.predict(total_data)
                print(res)
            
            elif mode == 'xgboost':
                res = xgboost.predict(total_data)
                print(res)
            
            elif mode == 'knn':
                res = knn.predict(total_data)
                print(res)
            print(ind[res[0]])
            return ind[res[0]]
        result = analysis(request.data)
        nu+=1
        return helloworld_pb2.PredictResult(res='Predict result is :' + result+str(nu))


def serve():    

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_MLpredictServicer_to_server(MLpredict(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
