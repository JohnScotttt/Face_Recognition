# 基于Arcface的人脸识别软件

## 介绍

这款软件旨在通过对比两张正面人脸图像的相似度，判断它们是否属于同一人。该系统基于深度学习和对比学习算法，能够实时评估两张照片是否为同一人物，无需依赖庞大的人脸数据库进行比对。模型采用对比学习的框架，在训练过程中每个batch使用11N结构，即一对同组正样本和N张随机选择的非同组负样本，从而实现无监督学习。骨干网络采用IResNet进行特征提取，并通过对比学习策略对输出的特征向量进行评估。为了提升判别精度，损失函数采用ArcCELoss。最终，系统通过计算两张图像特征向量之间的余弦相似度来判断它们的相似度，从而实现高效、准确的人脸匹配。

## usage

### environment

Python版本推荐使用3.10，使用以下命令安装基础依赖库

```
pip install -r requirement.txt
```

自行选择安装适合版本的PyTorch。

### train

1. 使用`utils/description.py`对数据集生成描述文件，每个数据集只要生成一次即可

   ```
   python utils/description.py /dataset /save_path
   ```

2. 修改`config`下的yml文件内容，或自行建立yml文件，调试符合要求的参数和超参数。

3. 命令行运行`train.py`进行训练

   ```
   python train.py YAML
   ```

### predict

使用训练好的pth进行预测，把pth文件路径添加到load_model后面，运行`pred.py`进行预测

```
python pred.py YAML IMG1 IMG2
```

其中YAML要与训练时的保持一致，IMG1和IMG2就是要判断的两张图片。

### config

config文件使用YAML格式，格式参考`config`下的demo yml文件

## 核心内容

