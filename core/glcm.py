import numpy as np
import cv2
import torch
from skimage.feature import graycomatrix, graycoprops
from utils import utils

# dim 72
def get_feature_path(path):
    features = []
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 读取灰度图像

    # 计算灰度共生矩阵，参数：图像矩阵，步长，方向，灰度级别，是否对称，是否标准化
    # [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4] 0、45、90、135度
    glcm = graycomatrix(image, [1, 2, 4], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
    # 循环计算表征纹理的参数 对比度、相异性、同质性、能量、自相关、ASM能量
    for prop in ['contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM']:
        temp = graycoprops(glcm, prop)
        temp = np.array(temp).reshape(-1)
        features.extend(temp)
    return torch.Tensor(features)

def get_feature_dir(dir):
    paths = utils.dir_2_filelist(dir)
    i = 0
    features = []
    length = len(paths)
    for path in paths:
        feature = get_feature_path(path)
        features.append((path.split("\\")[-1], feature))
        print(path.split("\\")[-1], "{} / {}".format(i, length))
        i += 1
    return features