from core import lbp, vgg16
from store import mysql
import cv2

HTTP = "http"
LOCAL = "local"

VGG16 = "vgg16"
LBP = "lbp"

FAISS = "faiss"
COSINE = "cosine"

class Search():
    # cp_mode, 比对方式，可选faiss或者torch cosine
    def __init__(self, mode = LOCAL, db = mysql.TestDB, cp_mode = COSINE) -> None:
        self.mode = mode
        if self.mode == LOCAL:
            self.feats_vgg16 = []
            self.feats_lbp = []

            self.db = mysql.DB(database = db)
            results = self.db.select_all()
            for r in results:
                self.feats_vgg16.append(r[3])
                self.feats_lbp.append(r[4])

            if cp_mode == FAISS:
                from . import faissdb
                self.engine_vgg16 = faissdb.FaissL2(self.feats_vgg16)
                self.engine_lbp = faissdb.FaissL2(self.feats_lbp)
            else:
                from . import cosine
                self.engine_vgg16 = cosine.Cosine(self.feats_vgg16)
                self.engine_lbp = cosine.Cosine(self.feats_lbp)
                    
        elif self.mode == HTTP:
            from . import http
            self.client = http.Clinet()

    def search(self, img_path, algorithm=VGG16):
        print("search mode:", self.mode)
        print("use algorithm:", algorithm)
        if self.mode == LOCAL:
            if algorithm == VGG16:
                return self.searcher(img_path, self.engine_vgg16, vgg16.get_feature_path)
            elif algorithm == LBP:
                return self.searcher(img_path, self.engine_lbp, lbp.get_feature_path)
        elif self.mode == HTTP:
            return self.client.search(img_path, algorithm)

    def searcher(self, img_path, engine, extract_func):
        scores, indexs = engine.search(extract_func(img_path))
        img_bufs = []
        for index in indexs:
            # 数据库中id是从1开始的，而比对引擎中的id是从0开始的
            path = self.db.get_one_path(index+1)
            img = cv2.imread(path)
            img_bufs.append(img)  # 返回图像路径
        return scores, img_bufs