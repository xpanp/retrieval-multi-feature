# retrieval-multi-feature

Image retrieval using multiple features

使用多特征进行图像检索

## Introduction

- 该项目使用多种算法提取图像特征，用于图像检索。特征包括vgg16网络提取的特征、纹理特征、颜色直方图特征、glcm的统计特征以及vit网络。希望通过对比不同特征用于图像检索，比较其检索效果，探讨图像特征表达的相关问题。

- 使用[Faiss](https://faiss.ai/)进行特征向量比对的加速。作为对比，也可使用`pytorch`自带的`cosine_similarity`函数进行比较，详情见`search/cosine.py`。

- 使用`MySQL`存储原始图片信息以及提取的各类原始特征。该场景下更适合`nosql`数据库, 使用`MySql`仅出于学习使用的考虑。

- 目前使用C/S架构，客户端使用PyQT，客户端与服务端使用HTTP进行通讯。也可以仅使用client启动，将其启动模式设置为`local`。

## Deploy

### Faiss

使用`conda`安装`faiss`

```bash
conda install openblas
conda install faiss-cpu -c pytorch
```

### MySQL

使用`docker`安装`mysql`

```bash
docker pull mysql:latest
docker run --restart=always -itd --name mysql-test -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 mysql
```

## Build Image Dataset

将需要建库的图片一次性放在文件夹下，运行如下命令开始建库：

```shell
python get_feature.py --datadir=~/image-test --db_host=127.0.0.1 --db_user=admin --db_passwd=admin --db_database=test
```

该过程较长，建议后台运行。

## RUN

### 仅客户端模式启动

```shell
python client.py --datadir=~/image-test --db_host=127.0.0.1 --db_user=admin --db_passwd=admin --db_database=test --mode=local
```

### 以C/S架构启动

服务端

```shell
python server.py --datadir=~/image-test --db_host=127.0.0.1 --db_user=admin --db_passwd=admin --db_database=test
```

客户端

```shell
python client.py --mode=http
```

## TODO

1. 特征融合
2. 改为B/S架构。
3. 图像传输需要压缩。
4. 更换数据库。
5. 将建库功能加入server端。
6. 思考关于特征融合、编码、检索、相关度、重排的问题及实现。

## Reference

[策略算法工程师之路-基于内容的图像检索(CBIR)](https://zhuanlan.zhihu.com/p/158740736)

[实现vgg16特征图可视化、图像卷积运算数据流向解析（pytorch）](https://blog.csdn.net/qq_44442727/article/details/112977805)

[Python-Image-feature-extraction](https://github.com/1044197988/Python-Image-feature-extraction.git)
