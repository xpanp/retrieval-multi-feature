import time
import requests
import base64
import json
import numpy as np
import cv2
from urllib.parse import urljoin

from . import search

class Clinet():
    def __init__(self, args) -> None:
        self.url = 'http://' + args.host + ':' + str(args.port)
        print('target url:', self.url)

    def search(self, filepath, algorithm=search.VGG16, top_k=4):
        url = urljoin(self.url, '/search')
        file = open(filepath, 'rb')
        # 拼接参数
        files = {'file': (filepath.split('/')[-1], file, 'image/jpg')}
        datas = {'algo': algorithm}
        # 发送post请求到服务器端
        t1 = time.time()
        r = requests.post(url, files=files, data=datas)
        t2 = time.time()
        print('特征搜索时间:%s毫秒' % ((t2 - t1)*1000))
        json_datas = json.loads(r.text)
        img_bufs = []
        scores = []
        for i in range(len(json_datas)):
            res = json.loads(json_datas[i])
            img = base64.b64decode(res["img"], altchars=None, validate=False)  # 将图片转换为str格式
            img = np.asarray(bytearray(img), dtype="uint8")  # 将图片转换为numpy buf格式
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img_bufs.append(img)
            scores.append(res['score'])
        return scores, img_bufs
