import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import torch
from utils import utils

# 获取模型以及图片预处理handle
def get_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform

# 从图片路径中提取特征 torch.Size([768])
def get_feature(path, model, transform):
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model.forward_features(tensor)
    out = torch.squeeze(out)
    return out

# 输入文件夹路径，提取特征
# 返回结果list，(图片名称, 特征) torch.Size([768])
def get_feature_dir(d):
    model, transform = get_model()

    paths = utils.dir_2_filelist(d)
    features = []
    length = len(paths)
    i = 0
    for path in paths:
        feature = get_feature(path, model, transform)
        features.append((path.split("\\")[-1], feature))
        print(path.split("\\")[-1], "{} / {}".format(i, length))
        i += 1
    return features

# 输入图片路径，提取特征 torch.Size([768])
def get_feature_path(path):
    model, transform = get_model()
    return get_feature(path, model, transform)
