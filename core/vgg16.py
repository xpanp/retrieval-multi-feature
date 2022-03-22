from utils import utils
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# TODO 使用自己构造的cnn提取特征

# 将输入图片转化为卷积运算格式tensor数据
def transfer_image(image_path):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_path).convert('RGB')
    # 数据预处理方法
    # TODO 不可只取中间部分
    image_transform = transforms.Compose([
        # 将图像最小边缩放至224,长边同比例缩放
        transforms.Resize(224),
        # 从图象中间剪切224*224大小
        transforms.CenterCrop(224),
        # 将图像数据转化为tensor
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        # std=[0.229, 0.224, 0.225]),    #图像归一化??? 参数如何得到
    ])
    # 更新载入图像的数据格式
    image_info = image_transform(image_info)
    # 增加tensor维度，便于卷积层进行运算
    image_info = image_info.unsqueeze(0)
    return image_info


# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        # feature_extractor是特征提取层，后面可以具体看一下vgg16网络
        for index, layer in enumerate(feature_extractor):
            # x是输入图像的张量数据，layer是该位置进行运算的卷积层，就是进行特征提取
            x = layer(x)
            # k代表想看第几层的特征图
            if k == index:
                return x


# 可视化特征图
def show_feature_map(feature_map):
    # squeeze(0)实现tensor降维，开始将数据转化为图像格式显示
    feature_map = feature_map.squeeze(0)
    # 进行卷积运算后转化为numpy格式
    feature_map = feature_map.cpu().numpy()
    # 特征图数量等于该层卷积运算后的特征图维度
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
    plt.show()


# 从图片路径中提取特征，需初始化特征提取器
def get_feature(path, feature_extractor, use_gpu):
    image_info = transfer_image(path)
    if use_gpu:
        image_info = image_info.cuda()

    # 提取第28层的特征图(第29层开始是全连接层)
    feature_map = get_k_layer_feature_map(feature_extractor, 28, image_info)
    # vgg16第28层为512*14*14,全局平均池化，提取512维特征
    avg_pool = nn.AvgPool2d(14, stride=1)
    feature_map_avg = avg_pool(feature_map)
    # 降维
    feature_map_avg_s = torch.squeeze(feature_map_avg)
    return feature_map_avg_s


# 输入文件夹路径，提取特征
# 返回结果list，(图片名称, 特征)
def get_feature_dir(d):
    # 导入Pytorch封装的vgg16网络模型
    model = models.vgg16(pretrained=True)
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        model = model.cuda()
    feature_extractor = model.features

    paths = utils.dir_2_filelist(d)
    features = []
    length = len(paths)
    i = 0
    for path in paths:
        feature = get_feature(path, feature_extractor, use_gpu)
        features.append((path.split("\\")[-1], feature))
        print(path.split("\\")[-1], "{} / {}".format(i, length))
        i += 1
    return features


# 输入图片路径，提取特征
def get_feature_path(path):
    # 导入Pytorch封装的vgg16网络模型
    model = models.vgg16(pretrained=True)
    # 是否使用gpu运算
    # use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        model = model.cuda()
    feature_extractor = model.features

    feature = get_feature(path, feature_extractor, use_gpu)
    return feature
