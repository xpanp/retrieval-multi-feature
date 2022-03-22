import time
import requests
import base64
import json
import numpy as np
import cv2

from . import search

class Clinet():
    def __init__(self, url = "http://10.199.130.68:11820/search") -> None:
        self.url = url
        pass

    def search(self, filepath, algorithm=search.VGG16, top_k=4):
        file = open(filepath, 'rb')
        # 拼接参数
        files = {'file': (filepath.split('/')[-1], file, 'image/jpg')}
        # 发送post请求到服务器端
        t1 = time.time()
        r = requests.post(self.url, files=files)
        t2 = time.time()
        print('特征搜索时间:%s毫秒' % ((t2 - t1)*1000))
        json_datas = json.loads(r.text)
        img_bufs = []
        for i in range(len(json_datas)):
            res = json.loads(json_datas[i])
            img = base64.b64decode(res["img"], altchars=None, validate=False)  # 将图片转换为str格式
            img = np.asarray(bytearray(img), dtype="uint8")  # 将图片转换为numpy buf格式
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img_bufs.append(img)
        return [], img_bufs
