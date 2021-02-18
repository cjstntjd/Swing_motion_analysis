from __future__ import print_function
import logging

import grpc

import os

import helloworld_pb2
import helloworld_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    #
    # For more channel options, please see https://grpc.io/grpc/core/group__grpc__arg__keys.html
    with grpc.insecure_channel(target='localhost:50051',
                               options=[('grpc.lb_policy_name', 'pick_first'),
                                        ('grpc.enable_retries', 0),
                                        ('grpc.keepalive_timeout_ms', 10000)
                                       ]) as channel:
        stub = helloworld_pb2_grpc.MLpredictStub(channel)
        f = open('/home/younghwan/swing_analysis/swing/back_cut/EXER_20210109_115203.csv','r')
        data = f.read()
        # Timeout in seconds.
        # Please refer gRPC Python documents for more detail. https://grpc.io/grpc/python/grpc.html
        li = ['lstm','svm','knn','nb','rf','xgboost']
        for mode in li:
            tmp = '|'
            tmp = mode + tmp
            tmp+=data
            response = stub.Predict(helloworld_pb2.DataPredict(data=tmp),
                                    timeout=10)
            print("Greeter client received: " + response.res)


if __name__ == '__main__':
    logging.basicConfig()
    run()

