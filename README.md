# 基于Arcface的人脸识别软件

## 介绍

这款软件旨在通过对比两张正面人脸图像的相似度，判断它们是否属于同一人。该系统基于深度学习和对比学习算法，能够实时评估两张照片是否为同一人物，无需依赖庞大的人脸数据库进行比对。模型采用对比学习的框架，在训练过程中每个组使用11N结构，即一对同组正样本和N张随机选择的非同组负样本，从而实现无监督学习。骨干网络采用IResNet进行特征提取，并通过对比学习策略对输出的特征向量进行评估。为了提升判别精度，损失函数采用ArcCELoss。最终，系统通过计算两张图像特征向量之间的余弦相似度来判断它们的相似度，从而实现高效、准确的人脸匹配。

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

Config文件使用YAML格式，格式参考`config`下的demo yml文件

## 核心内容

由于采用对比学习的方式，每一组数据应该包含一对同标签的正样本对和n个与正样本不同标签的负样本，组成11N的结构。模型训练时将这样的一组数据作为最小单位，取一个`batch_size`大小的数据输入到网络当中，输入尺寸为$[bs \times sub\_bs , c , h , w]$其中$sub\_bs=2+n$，输出尺寸为$[bs , sub\_bs , dims]$。对于每一组数据，取第一个向量为基准，求剩下的向量与该向量的Arc余弦相似度为logits。由于一组数据中第一、二个向量取自同标签的不同图，其余为不同标签的图，故对于一组数据来说其label始终是0。对logits和label求CossEntropyLoss即完成了整个forward过程。

### Dataloader

Dataloader使用torch.utils.data的Dataset和Dataloader框架。定义超参数`neg_batch_size`为负样本个数n。定义描述文件description.txt格式为：原始标签\t图片路径\n。

FRdataset首先读取description，创建一个字典，其key为原始标签，value为对应标签所有图片路径构成的数组。读取完description所有内容后，将字典的value重排成数组，实现去除原始标签，将其定义为images_path_labels_list。`__gititem__`方法取images_path_labels_list中下标为idx的数组为whole_pos_list，从whole_pos_list随机取两个值作为pos_pairs，去除images_path_labels_list中下标为idx的数组，剩下的整合成一维数组定义为whole_neg_list，取whole_neg_list随机`neg_batch_size`个值为neg_path_list，合并pos_pairs和neg_path_list为一个数组group，用定义的方法`read_list_img`读取数组内所有路径为图片数组，再将图片数组通过transforms处理后打包成batch。

Dataloader使用torch原生类，使用了`batch_size`、`shuffle`和`num_workers`超参。

### Net

由于是对比学习，所以没有固定的网络模型，baseline采用的是IResNet50作为骨干网络，`FRbackbone.py`还有['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200', 'resnet18', 'resnet50', 'resnet101', 'resnet152']预设提供使用。预设的forward采用torch.cuda.amp.autocast(self.fp16)混合精度减少开支。

### Loss function

Loss函数采用基于ArcFace的ArcCELoss，继承自torch.nn.Module，定义超参数`s`为尺度因子 (Scaling factor)，定义超参数`margin`为边际损失 (Margin for the decision boundary)。同时`FRLoss.py`提供传统的余弦相似度损失函数mmCELoss（不推荐）。ArcFace的相关知识可以参考Deng J, Guo J, Xue N, et al. Arcface: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 4690-4699.

对于每一组数据，取第一个向量为query，剩下的为key，求query与key的Arc余弦相似度为logits。对logits和label求CossEntropyLoss完成Loss函数过程。
