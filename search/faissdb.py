import faiss
import numpy as np
from abc import abstractmethod, ABCMeta


class Faiss(metaclass=ABCMeta):
    @abstractmethod
    # feats type is list
    def __init__(self, feats, dim=512) -> None:
        pass

    @abstractmethod
    # feat type is tensor, return scores as numpy.ndarray, numpy.float32
    def search(self, feat, top_k=4):
        pass


# 精确搜索 欧氏距离
class FaissL2(Faiss):
    def __init__(self, feats, dim=512) -> None:
        self.index = faiss.IndexFlatL2(dim)
        vec_array = np.array(feats).astype('float32')
        self.index.add(vec_array)

    def search(self, feat, top_k=4):
        D, I = self.index.search(np.array([feat.tolist()]).astype('float32'), top_k)
        return D[0], I[0]


# IndexIVFFlat 更快的搜索
class FaissLVF(Faiss):
    def __init__(self, feats, dim=512) -> None:
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, 5, faiss.METRIC_L2)
        vec_array = np.array(feats).astype('float32')
        self.index.train(vec_array)
        self.index.add(vec_array)

    def search(self, feat, top_k=4):
        D, I = self.index.search(np.array([feat.tolist()]).astype('float32'), top_k)
        return D[0], I[0]


# IndexIVFPQ 更低的内存占用，非精确搜索，大量数据时可用，至少200张图片
class FaissLVFPQ(Faiss):
    def __init__(self, feats, dim=512) -> None:
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, 5, 4, 8)
        vec_array = np.array(feats).astype('float32')
        self.index.train(vec_array)
        self.index.add(vec_array)

    def search(self, feat, top_k=4):
        D, I = self.index.search(np.array([feat.tolist()]).astype('float32'), top_k)
        return D[0], I[0]
