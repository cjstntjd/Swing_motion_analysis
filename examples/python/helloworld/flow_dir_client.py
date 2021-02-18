from __future__ import print_function
import logging

import grpc

import os

import helloworld_pb2
import helloworld_pb2_grpc
import time

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    #
    # For more channel options, please see https://grpc.io/grpc/core/group__grpc__arg__keys.html
    with grpc.insecure_channel(target='125.136.42.188:5001',
                               options=[('grpc.lb_policy_name', 'pick_first'),
                                        ('grpc.enable_retries', 0),
                                        ('grpc.keepalive_timeout_ms', 10000)
                                       ]) as channel:
        stub = helloworld_pb2_grpc.MLpredictStub(channel)
        file_name_dir = []
        dir_location = '/home/younghwan/swing_analysis/swing'
        for root , dirs , files in os.walk(dir_location):
            for fname in files:
                full_fname = os.path.join(root,fname)
                file_name_dir.append(full_fname)
        n=0
        now = time.time()
        for file_input in file_name_dir:        
            f = open(file_input,'r')
            data = f.read()
        # Timeout in seconds.
        # Please refer gRPC Python documents for more detail. https://grpc.io/grpc/python/grpc.html
            response = stub.Predict(helloworld_pb2.DataPredict(data=data),
                                 timeout=10)
            print("Greeter client received: " + response.res +'    '+ str(n))
            n+=1
    print('final_time : '+ str(time.time() - now) )

if __name__ == '__main__':
    logging.basicConfig()
    run()

