# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

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
global model
global graph
graph = tf.get_default_graph()
model = load_model('/home/younghwan/swing_analysis/model_x.h5')
model.load_weights('/home/younghwan/swing_analysis/model_x_weights.h5')
model._make_predict_function()
global power
global mode
mode = 'knn'
po_li = ['lstm','knn','rf','xgboost']
if mode in po_li:
    power = 4
else:
    power = 1




class MLpredict(helloworld_pb2_grpc.MLpredictServicer):

    def Predict(self,request,context):
        def analysis(d):
            
            tar_li = ['AX','AY','AZ']
            ind = ['back_cut', 'back_drive', 'back_short', 'back_smash', 'fo_cut' ,'fo_drive',
    'fo_short' ,'fo_smash']
            data = d.split('\n')
            data.pop(0)
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
                tmp.append((df['AX'][i]/1000)**power)
                tmp.append((df['AY'][i]/1000)**power)
                tmp.append((df['AZ'][i]/1000)**power)
                tmp_li.append(tmp)
            total_data = []
            total_data.append(tmp_li)
            total_data = np.array(total_data)
            

            with graph.as_default():
                set_session(session)
                res = model.predict(total_data)
            res = np.argmax(res,axis=-1)
            return ind[res[0]]
        result = analysis(request.data)
        return helloworld_pb2.PredictResult(res='Predict result is : %s' % result)


def serve():    

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_MLpredictServicer_to_server(MLpredict(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    
    serve()
