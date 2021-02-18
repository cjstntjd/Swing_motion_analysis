from concurrent import futures
import logging
import os
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

global df_data
df_data = pd.read_csv('/home/younghwan/userdata/user_data.csv',index_col=0)
global df_index
df_index = ['user_id','name','ai_num','back_cut','back_drive','back_short','back_smash','fo_cut','fo_drive','fo_short','fo_smash']
global sync_index
sync_index = ['back_cut','back_drive','back_short','back_smash','fo_cut','fo_drive','fo_short','fo_smash']

global model_list
model_list = [lstm,rf,xgboost,nb,svm,knn]
print(df_data.head())
class MLpredict(helloworld_pb2_grpc.MLpredictServicer):

    def Predict(self,request,context):
        
        def recog_siganl(signal):
            global df_data
            split_signal = signal.split('|')
            tmp_id = split_signal[0]
            tmp_mode = split_signal[1]
            tmp_model = split_signal[2]
            tmp_data = split_signal[3]
            print('get : ',signal)
            #닉네임 설정 모드 
            if tmp_mode == '0' and (tmp_id not in list(df_data['user_id'])):
                #first come and no tutorial -> real newb
                
                if tmp_data in list(df_data['name']):
                    ret = tmp_id+'|'+tmp_mode+'|'+tmp_model+'|'+'duplicate'
                    return ret
                elif tmp_data not in list(df_data['name']):
                    user_template = []
                    user_template.append(tmp_id)
                    user_template.append(tmp_data)
                    user_template.append(tmp_model)
                    user_template.extend([0,0,0,0,0,0,0,0])
                    df_data = df_data.append(pd.Series(user_template,index = df_index),ignore_index=True)
                    ret = tmp_id+'|'+ '1' +'|'+tmp_model+'|'+'complete' # go to Tutorial
                    print(df_data.head())
                    return ret
            

            # 튜토리얼 모드
            elif tmp_mode == '1' and tmp_model == '0':
                #now client make their own name but he can't do Tutorial
                #make stream pipline
                tar_li = ['AX','AY','AZ']
                swing_index = ['back_cut', 'back_drive', 'back_short', 'back_smash', 'fo_cut' ,'fo_drive','fo_short' ,'fo_smash']
                
                data = tmp_data.split('\n')
                index = data.pop(0)
                index = index.split(',')
                tmp_real_data = []

                for dat_num in range(len(data)):
                    if data[dat_num] == '':
                        continue
                    tmp_real_data.append(data[dat_num].split(','))
                
                df = pd.DataFrame(tmp_real_data,columns=index)

                for y in index:
                    df[y] = pd.to_numeric(df[y],downcast='float')
                
                model_power = [4,4,4,1,1,2]
                model_name = [lstm,rf,xgboost,nb,svm,knn]
                result_string = ''
                for num in range(len(model_power)):
                    tmp_li = []
                    for i in range(len(df)):
                        tmp = []
                        for tmp_ax in tar_li:
                            tmp.append((df[tmp_ax][i]/1000)**model_power[num])
                        tmp_li.append(tmp)

                    predict_data = []
                    predict_data.append(tmp_li)
                    predict_data = np.array(predict_data)

                    #여기서부터 모델에 대한 다른 형식으로 지정이 필요하게 됨
                    #위의 power 부분은 순차적으로 이동하는 형태 이므로 시작 시점 num == 0 인 경우에 한해서 reshape를 안하는게 좋을것 같음

                    if num == 0 : 
                        #mode 가 장단기 기억에 국한 되는 경우에 한해서 입력 텐서 재조정 하지 않을 것 
                        with graph.as_default():
                            set_session(session)
                            res = model_name[num].predict(predict_data)
                        res = np.argmax(res,axis=-1)
                    else: # 장단기 기억이 아닌 경우 only
                        nsamples,nx,ny = predict_data.shape
                        predict_data = predict_data.reshape((nsamples,nx*ny))
                        #입력 행렬 재조정
                        res = model_name[num].predict(predict_data)
                    result_string = result_string + str(res[0]) + ','
                n_ret = tmp_id+'|'+ '1' +'|'+tmp_model+'|'+ result_string[:-1]
                return n_ret
            
            elif tmp_mode == '1' and tmp_model != '0':
                # 모델이 선정되어 결과를 반환 하는 경우에 -> 데이터 프레임에 새 칼럼 추가 및 저장
                df_data['ai_num'][df_data['user_id']==tmp_id] = tmp_model
                ret = tmp_id + '|' + '1' + '|' + tmp_model + '|' + 'update_user_profile_complete'
                return ret
            
            elif tmp_mode == '2':
                #동기화 코드 작성
                sync_data = tmp_data.split(',')
                for x in range(len(sync_index)):
                    df_data[sync_index[x]][df['user_id'] == tmp_id] = int(sync_data[x])
                    #원래 되돌려 줄때는 상위 몇명 보내줘야 되는데 아직은 미구현 
                
                return tmp_id + '|' + tmp_mode + '|' + tmp_model + '|' + 'sync_complete'
            
            elif tmp_mode == '3':
                #연습 모드 코드 작성
                #에러 분류 하고 에러 코드 출력 하게 코드 작성해야함
                data = tmp_data.split('\n')
                index = data.pop(0)
                index = index.split(',')
                tmp_real_data = []

                for dat_num in range(len(data)):
                    if data[dat_num] == '':
                        continue
                    tmp_real_data.append(data[dat_num].split(','))

                df = pd.DataFrame(tmp_real_data,columns=index)

                for y in index_li:
                    df[y] = pd.to_numeric(df[y],downcast='float')
                tmp_li = []
                for i in range(len(df)):
                    tmp = []
                    tmp.append((df['AX'][i]))
                    tmp.append((df['AY'][i]))
                    tmp.append((df['AZ'][i]))
                    tmp_li.append(tmp)
                total_data = []
                total_data.append(tmp_li)
                total_data = np.array(total_data)

                if tmp_model != '1':
                    nsamples,nx,ny = total_data.shape
                    total_data = total_data.reshape((nsamples,nx*ny))

                if tmp_model == '1':
                    with graph.as_default():
                        set_session(session)
                        res = lstm.predict(total_data)
                    res = np.argmax(res,axis=-1)
                
                elif tmp_model == '2':
                    res = rf.predict(total_data)
                    print(res)

                elif tmp_model == '3':
                    res = xgboost.predict(total_data)
                    print(res)

                elif mode == '4':
                    res = nb.predict(total_data)
                    print(res)

                elif mode == '5':
                    res = svm.predict(total_data)
                    print(res)

                elif mode == '6':
                    res = knn.predict(total_data)
                    print(res)
                print(sync_index[res[0]])
                return tmp_id + '|' + '3' + '|' + tmp_model + '|' + sync_index[res[0]]

        result = recog_siganl(request.data)

        return helloworld_pb2.PredictResult(res=result)



def serve():    

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_MLpredictServicer_to_server(MLpredict(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()






if __name__ == '__main__':
    logging.basicConfig()
    serve()
