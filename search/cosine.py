import torch
import time

# 获取列表的第二个元素
def take_second(elem):
    return elem[1]

# 比较两个tensor，返回相似度得分
def compare(k, f):
    return torch.cosine_similarity(k, f, dim=0)

class Cosine():
    # feats type is list
    def __init__(self, feats, dim=0) -> None:
        self.feats = torch.tensor(feats)

    # feat type is tensor, return scores as [tensor, ]
    def search(self, feat, top_k=4):
        results = []
        for i in range(len(self.feats)):
            score = compare(feat, self.feats[i])
            results.append((i, score))
        results.sort(key=take_second, reverse=True)
        D = []
        I = []
        res_num = min(top_k, len(results))
        for i in range(res_num):
            I.append(results[i][0])
            D.append(results[i][1])
        return D, I
