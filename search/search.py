from core import lbp, vgg16
from . import cosine, faissdb
from store import mysql

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
                self.feats_lbp.append(r[3])

            if cp_mode == FAISS:
                self.engine_vgg16 = faissdb.FaissL2(self.feats_vgg16)
                self.engine_lbp = faissdb.FaissL2(self.feats_lbp)
            else:
                self.engine_vgg16 = cosine.Cosine(self.feats_vgg16)
                self.engine_lbp = cosine.Cosine(self.feats_lbp)
            
        
        elif self.mode == HTTP:
            pass

    def search(self, img_path, algorithm=VGG16):
        print("search mode:", self.mode)
        print("use algorithm:", algorithm)
        if self.mode == LOCAL:
            if algorithm == VGG16:
                return self.search_engine(img_path, self.engine_vgg16, vgg16.get_feature_path)
            elif algorithm == LBP:
                return self.search_engine(img_path, self.engine_lbp, lbp.get_feature_path)
        elif self.mode == HTTP:
            pass

    # TODO 直接返回图像二进制，而不是返回文件路径
    def search_engine(self, img_path, engine, extract_func):
        D, I = engine.search(extract_func(img_path))
        N = []
        for index in I:
            # 数据库中id是从1开始的，而比对引擎中的id是从0开始的
            r = self.db.select_one(index+1) 
            N.append(r[2])  # 返回图像路径
        return D, N