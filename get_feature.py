from core import extract, vgg16, lbp
from utils import utils
from store import mysql
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # dir
    parser.add_argument('--datadir', type=str, default='~',
        help='dataset dir')

    # database
    parser.add_argument('--db_host', type=str, default='127.0.0.1',
        help='database host')

    parser.add_argument('--db_user', type=str, default='admin',
        help='database user name')

    parser.add_argument('--db_passwd', type=str, default='admin',
        help='database user passwd')

    parser.add_argument('--db_database', type=str, default='test',
        help='database name')

    args = parser.parse_args()

    return args

# 从文件夹中使用vgg16依次提取图像特征并存入文件
def get_feature_vgg16(image_dir):
    features = vgg16.get_feature_dir(image_dir)
    torch.save(features, "data/features-pattern.pt")

# 从文件夹中使用lbp依次提取图像特征并存入文件
def get_feature_lbp(image_dir):
    features = lbp.get_feature_dir(image_dir)
    torch.save(features, "data/features-voc-lbp.pt")

# 从文件夹中提取图像综合信息
def get_feature_dir(d):
    paths = utils.dir_2_filelist(d)
    results = []
    e = extract.Extractor()
    for i, path in enumerate(paths):
        out = e.get(path)
        results.append((i,) + out)
    return results

# 生成特征数据文件
if __name__ ==  '__main__':
    args = get_args()
    db = mysql.DB(args)
    results = get_feature_dir(args.datadir)
    # 将结果插入数据库
    for f in results:
        db.insert(f[0], f[1], f[2], f[3].tolist(), 
            f[4].tolist(), f[5].tolist(), f[6].tolist(), f[7].tolist())
    db.close()