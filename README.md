# retrieval-multi-feature

Image retrieval using multiple features

使用多特征进行图像检索

## Introduction

- 该项目使用多种算法提取图像特征，用于图像检索。特征包括vgg16网络提取的特征、纹理特征以及其他特征(待添加)。希望通过对比不同特征用于图像检索，比较其检索效果，探讨图像特征表达的相关问题。

- 使用[Faiss](https://faiss.ai/)进行特征向量比对的加速。作为对比，也可使用`pytorch`自带的`cosine_similarity`函数进行比较，详情见`search/cosine.py`。

- 使用MySQL存储原始图片信息以及提取的各类原始特征。

- 目前使用C/S架构，客户端使用PyQT，客户端与服务端使用HTTP进行通讯。

## Deploy

### Faiss

使用`conda`安装

```bash
conda install openblas
conda install faiss-cpu -c pytorch
```

### MySQL

使用`docker`安装

```bash
docker pull mysql:latest
docker run --restart=always -itd --name mysql-test -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 mysql
```

## TODO

1. 增加配置文件。
2. 部署文档，如客户端部署，服务端部署，服务端训练。
3. 增加更多的传统特征提取算法，如颜色特征、glcm特征。
4. 更换vgg16网络，可尝试其他网络。
5. 改为B/S架构。
6. 图像传输需要压缩。
7. 思考关于特征融合、编码、检索、相关度、重排的问题及实现。

## Reference

[策略算法工程师之路-基于内容的图像检索(CBIR)](https://zhuanlan.zhihu.com/p/158740736)

[实现vgg16特征图可视化、图像卷积运算数据流向解析（pytorch）](https://blog.csdn.net/qq_44442727/article/details/112977805)

[Python-Image-feature-extraction](https://github.com/1044197988/Python-Image-feature-extraction.git)
