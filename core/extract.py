from . import vgg16, lbp, color, glcm, vit
import torch
from torchvision import models

# 特征提取器
class Extractor():
    def __init__(self) -> None:
        # 导入Pytorch封装的vgg16网络模型
        vgg_model = models.vgg16(pretrained=True)
        self.vgg_feature_extractor = vgg_model.features
        self.lbp = lbp.LBP()
        self.vit_model, self.vit_trans = vit.get_model()
    
    # 提取vit特征
    def get_vit_feature(self, path):
        return vit.get_feature(path, self.vit_model, self.vit_trans)

    # 提取vgg16特征
    def get_vgg16_feature(self, path):
        return vgg16.get_feature(path, self.vgg_feature_extractor, False)

    # 提取lbp特征
    def get_lbp_feature(self, path):
        image_array = self.lbp.describe(path)
        basic_array = self.lbp.lbp_basic(image_array)
        hist = self.lbp.get_hist(basic_array, [256], [0, 256])
        thist = torch.from_numpy(hist)
        return thist

    def get_color_feature(self, path):
        return color.get_feature_path(path)

    def get_glcm_feature(self, path):
        return glcm.get_feature_path(path)

    # 获取综合信息及特征
    def get(self, path):
        return (path.split("\\")[-1], path, self.get_vgg16_feature(path), 
            self.get_lbp_feature(path), self.get_color_feature(path), 
            self.get_glcm_feature(path), self.get_vit_feature(path))