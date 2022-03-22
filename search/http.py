# import time
# import requests
# import base64
# import json
# import numpy as np

# TODO 待做

# url = "http://10.199.130.254:11820/search"

# def search(ui):
#     print("待搜索图片：", self.img_path)
#     file_name = self.img_path.split('/')[-1]
#     file = open(self.img_path, 'rb')
#     # 拼接参数
#     files = {'file': (file_name, file, 'image/jpg')}
#     # 发送post请求到服务器端
#     t1 = time.time()
#     r = requests.post(url, files=files)
#     t2 = time.time()
#     print('特征搜索时间:%s毫秒' % ((t2 - t1)*1000))
#     json_datas = json.loads(r.text)
#     img_res = []
#     for i in range(len(json_datas)):
#         res = json.loads(json_datas[i])
#         img = base64.b64decode(res["img"], altchars=None, validate=False)  # 将图片转换为str格式
#         img = np.asarray(bytearray(img), dtype="uint8")  # 将图片转换为numpy buf格式
#         img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#         img_res.append(img)
#         # cv2.imshow('{}.jpg'.format(i),img)
#     ui.label_1.setPixmap(img_norm(img_res[0]))
#     ui.label_2.setPixmap(img_norm(img_res[1]))
#     ui.label_3.setPixmap(img_norm(img_res[2]))
#     ui.label_4.setPixmap(img_norm(img_res[3]))