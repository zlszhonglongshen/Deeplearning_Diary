# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 2020/05/22 15:57
@Author  : Johnson
@Email   : 593956670@qq.com
"""
bs("./test/IMG_4860.JPG")

import base64
def bs(io):
    with open(io,'rb') as f:base64_data = base64.b64encode(f.read())
    import grpc
    import calculate_pb2
    import calculate_pb2_grpc

    # 打开 gRPC channel，连接到 localhost:50051
    channel = grpc.insecure_channel('localhost:5566')
    # 创建一个 stub (gRPC client)
    stub = calculate_pb2_grpc.CalculateStub(channel)
    # 创建一个有效的请求消息 Number
    number = calculate_pb2.Number(value=base64_data)
    # 带着 Number 去调用 Square
    response = stub.Square(number)
    return  (response.value)

from tqdm import tqdm
for i in tqdm(range(1000)):
    a = bs("./test/w30.jpg")
    print(a)

from tqdm import tqdm
for i in tqdm(range(1000)):
    a = bs("062.jpg")
    print(a)

import concurrent
executor_pool = concurrent.futures.ProcessPoolExecutor(10)
import time
def test():
    tim1 = time.time()
    a = bs("./test/w1.jpg")
    tim2 = time.time()
    print("shijain",tim2-tim1)
    print(a)
futures = []
for i in (range(1000)):
    future = executor_pool.submit(test)
    futures.append(future)

for f in futures:
    f.result()



