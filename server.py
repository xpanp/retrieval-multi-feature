from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
import os
import base64
import json

from search import search

class SearchServer:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = '.'

    def __init__(self, db_name) -> None:
        self.app.add_url_rule('/upload',endpoint='/upload',view_func=self.upload_file)
        self.app.add_url_rule('/search',endpoint='/search',view_func=self.search, methods=['POST'])
        
        self.search_engine = search.Search(mode = search.LOCAL, db = db_name, cp_mode=search.FAISS)
    
    # @app.route('/upload')
    def upload_file(self):
        return render_template('./upload.html')

    def search(self):
        f_form = request.files['file']
        filePath = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_filename(f_form.filename))
        f_form.save(filePath)
        scores, names = self.search_engine.search(filePath, search.VGG16)
        os.remove(filePath) # 删除搜索文件
        result = []
        for i in range(4):
            f = open(names[i], 'rb')
            b = base64.b64encode(f.read()).decode() #中文utf-8去掉b
            data = {"img": str(b), "score": float(scores[i])}
            json_str = json.dumps(data)
            result.append(json_str)
        return Response(json.dumps(result),  mimetype='application/json')

if __name__ == '__main__':
    s = SearchServer("imgfeatdb_test")
    s.app.run(host='0.0.0.0', port=11820, debug=True)