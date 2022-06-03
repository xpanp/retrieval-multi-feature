import cv2
import numpy as np
import torch
from utils import utils

# dim 9
def color_moments(path):
    img = cv2.imread(path)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.append(h_mean)
    color_feature.append(s_mean)
    color_feature.append(v_mean)
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.append(h_std)
    color_feature.append(s_std)
    color_feature.append(v_std)
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.append(h_thirdMoment)
    color_feature.append(s_thirdMoment)
    color_feature.append(v_thirdMoment)

    return torch.Tensor(color_feature)

# dim 256
def color_hist(path):
    img = cv2.imread(path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    thist = torch.from_numpy(hist)
    thist = torch.squeeze(thist, dim=1) # 降维
    return thist

def get_feature_path(path):
    return color_hist(path)

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