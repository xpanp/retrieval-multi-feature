from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
import os
import base64
import json
import argparse
import cv2
import numpy as np
import time

from search import search

def get_args():
    parser = argparse.ArgumentParser()

    # server
    parser.add_argument('--host', type=str, default='0.0.0.0',
        help='listen host')

    parser.add_argument('--port', type=int, default=11820,
        help='listen port')

    parser.add_argument('--debug', type=bool, default=False,
        help='debug')

    # searcher
    parser.add_argument('--datadir', type=str, default='~',
        help='dataset dir')

    parser.add_argument('--cp_mode', type=str, default='cosine',
        help='compare mode, support: cosine|faiss')

    # database
    parser.add_argument('--db_host', type=str, default='127.0.0.1',
        help='database host')

    parser.add_argument('--db_user', type=str, default='admin',
        help='database user name')

    parser.add_argument('--db_passwd', type=str, default='admin',
        help='database user passwd')

    parser.add_argument('--db_database', type=str, default='test',
        help='database name')

    args = parser.parse_args()

    return args

class SearchServer:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = '.'

    def __init__(self, args) -> None:
        self.app.add_url_rule('/upload',endpoint='/upload',view_func=self.upload_file)
        self.app.add_url_rule('/search',endpoint='/search',view_func=self.search, methods=['POST'])
        
        self.search_engine = search.Search(args, mode = search.LOCAL, cp_mode=args.cp_mode)
    
    # @app.route('/upload')
    def upload_file(self):
        return render_template('./upload.html')

    def search(self):
        algo = request.form['algo']
        f_form = request.files['file']
        filePath = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_filename(f_form.filename))
        f_form.save(filePath)
        t1 = time.time()
        scores, bufs = self.search_engine.search(filePath, algorithm=algo)
        t2 = time.time()
        print('特征搜索时间:%s毫秒' % ((t2 - t1)*1000))
        os.remove(filePath) # 删除搜索文件
        result = []
        for i in range(4):
            img_encode = cv2.imencode('.jpg', bufs[i])[1]
            b = base64.b64encode(img_encode).decode() #中文utf-8去掉b
            data = {"img": str(b), "score": float(scores[i])}
            json_str = json.dumps(data)
            result.append(json_str)
        return Response(json.dumps(result),  mimetype='application/json')

if __name__ == '__main__':
    args = get_args()
    s = SearchServer(args)
    s.app.run(host=args.host, port=args.port, debug=args.debug)