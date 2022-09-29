from core import lbp, vgg16, color, glcm, vit
from store import mysql
import cv2
from pathlib import Path

HTTP = "http"
LOCAL = "local"

VGG16 = "vgg16"
LBP = "lbp"
COLOR = "color"
GLCM = "glcm"
VIT = "vit"

FAISS = "faiss"
COSINE = "cosine"

class Search():
    def __init__(self, args, mode = LOCAL, cp_mode = COSINE) -> None:
        self.mode = mode
        if Path(args.datadir).exists():
            self.datadir = args.datadir
        elif mode == LOCAL:
            raise Exception("can not find data dir:", args.datadir)

        if self.mode == LOCAL:
            self.feats_vgg16 = []
            self.feats_lbp = []
            self.feats_color = []
            self.feats_glcm = []
            self.feats_vit = []

            self.db = mysql.DB(args)
            # 将特征加载到内存中，加快检索
            results = self.db.select_all()
            for r in results:
                self.feats_vgg16.append(r[3])
                self.feats_lbp.append(r[4])
                self.feats_color.append(r[5])
                self.feats_glcm.append(r[6])
                self.feats_vit.append(r[7])

            # cp_mode, 比对方式，可选faiss或者torch cosine
            if cp_mode == FAISS:
                from . import faissdb
                # TODO 将特征维度信息存到数据库中
                self.engine_vgg16 = faissdb.FaissL2(self.feats_vgg16, dim=512)
                self.engine_lbp = faissdb.FaissL2(self.feats_lbp, dim=256)
                self.engine_color = faissdb.FaissL2(self.feats_color, dim=256)
                self.engine_glcm = faissdb.FaissL2(self.feats_glcm, dim=72)
                self.engine_vit = faissdb.FaissL2(self.feats_vit, dim=768)
            else:
                from . import cosine
                self.engine_vgg16 = cosine.Cosine(self.feats_vgg16)
                self.engine_lbp = cosine.Cosine(self.feats_lbp)
                self.engine_color = cosine.Cosine(self.feats_color)
                self.engine_glcm = cosine.Cosine(self.feats_glcm)
                self.engine_vit = cosine.Cosine(self.feats_vit)
                    
        elif self.mode == HTTP:
            from . import http
            self.client = http.Clinet(args)

    def search(self, img_path, algorithm=VGG16):
        print("search mode:", self.mode)
        print("use algorithm:", algorithm)
        if self.mode == LOCAL:
            if algorithm == VGG16:
                return self.searcher(img_path, self.engine_vgg16, vgg16.get_feature_path)
            elif algorithm == LBP:
                return self.searcher(img_path, self.engine_lbp, lbp.get_feature_path)
            elif algorithm == COLOR:
                return self.searcher(img_path, self.engine_color, color.get_feature_path)
            elif algorithm == GLCM:
                return self.searcher(img_path, self.engine_glcm, glcm.get_feature_path)
            elif algorithm == VIT:
                return self.searcher(img_path, self.engine_vit, vit.get_feature_path)
        elif self.mode == HTTP:
            return self.client.search(img_path, algorithm)

    def searcher(self, img_path, engine, extract_func):
        scores, indexs = engine.search(extract_func(img_path))
        img_bufs = []
        for index in indexs:
            # 数据库中id是从1开始的，而比对引擎中的id是从0开始的
            path = self.db.get_one_name(index+1)
            path = str(Path(self.datadir).joinpath(path))
            img = cv2.imread(path)
            img_bufs.append(img)  # 返回图像路径
        return scores, img_bufs