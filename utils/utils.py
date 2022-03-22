import os

# 输入文件夹路径，返回文件列表，不可嵌套获取
def dir_2_filelist(d):
    names = os.listdir(d)
    paths = []  # 所有图片路径
    for name in names:
        paths.append(os.path.join(d, name))
    return paths