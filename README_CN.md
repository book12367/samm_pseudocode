# SAMM微表情识别算法伪代码

## 项目概述

本项目包含SAMM（Spontaneous Affective Facial Expressions in the Wild）微表情识别系统的完整算法伪代码。该系统使用面部关键点检测和深度学习技术来识别和分类微表情。

## 目录结构

```
pseudocode/
├── chinese/                 # 中文伪代码
│   ├── training/            # 训练算法
│   │   └── training_algorithm.txt
│   ├── inference/           # 推理算法
│   │   └── inference_algorithm.txt
│   ├── real_time_inference/ # 实时推理算法
│   │   └── real_time_algorithm.txt
│   └── preprocessing/       # 预处理算法
│       └── preprocessing_algorithm.txt
└── english/                 # 英文伪代码
    ├── training/            # 训练算法
    │   └── training_algorithm.txt
    ├── inference/           # 推理算法
    │   └── inference_algorithm.txt
    ├── real_time_inference/ # 实时推理算法
    │   └── real_time_algorithm.txt
    └── preprocessing/       # 预处理算法
        └── preprocessing_algorithm.txt
```

## 算法模块说明

### 1. 训练算法 (Training Algorithm)

训练算法实现了用于微表情识别的深度学习模型，主要特点包括：

- **特征提取**：使用MediaPipe面部网格检测面部关键点
- **序列对齐**：使用动态时间规整(DTW)对齐不同长度的视频序列
- **模型架构**：结合CNN和BiLSTM的混合架构，用于提取时空特征
- **数据增强**：包括时间扭曲和噪声添加等增强技术
- **评估指标**：在训练过程中计算召回率和F1分数

### 2. 推理算法 (Inference Algorithm)

推理算法用于对单个视频文件进行微表情分类：

- **视频处理**：逐帧提取面部关键点
- **序列处理**：对齐和标准化关键点序列
- **模型预测**：使用训练好的模型进行分类
- **结果输出**：返回预测类别、置信度和概率分布

### 3. 实时推理算法 (Real-time Inference Algorithm)

实时推理算法用于实时微表情检测：

- **缓冲区管理**：维护固定长度的帧缓冲区
- **实时处理**：连续处理摄像头输入
- **性能优化**：显示实时FPS以监控性能
- **用户界面**：提供可视化界面显示检测结果

### 4. 预处理算法 (Preprocessing Algorithm)

预处理算法用于数据准备和验证：

- **数据加载**：从Excel文件读取标签信息
- **文件验证**：检查视频文件的存在性和完整性
- **属性检查**：验证视频的帧数、分辨率等属性
- **数据清洗**：清理和标准化数据
- **格式转换**：将图像序列转换为视频格式

## 伪代码特点

- **非可执行性**：使用自定义语法避免直接运行
- **算法清晰**：保留完整的算法逻辑和流程
- **结构化设计**：采用面向对象和模块化设计
- **详细注释**：包含充分的注释说明算法细节
- **双语支持**：提供中文和英文两个版本

## 使用说明

每个算法文件都包含完整的实现细节，可以直接用于：

- 算法理解与学习
- 系统设计参考
- 学术研究支持
- 代码实现指导

## 技术栈

- **面部检测**：MediaPipe Face Mesh
- **深度学习**：TensorFlow/Keras
- **数据处理**：NumPy, Pandas
- **计算机视觉**：OpenCV
- **序列对齐**：FastDTW

## 算法流程

1. **数据预处理**：加载和验证SAMM数据集
2. **特征提取**：提取面部关键点序列
3. **序列对齐**：使用DTW对齐不同长度序列
4. **模型训练**：训练CNN-BiLSTM混合模型
5. **模型推理**：对新数据进行微表情分类
6. **结果评估**：计算分类准确率和相关指标

## 参考

- SAMM数据集：Spontaneous Affective Facial Expressions in the Wild
- MediaPipe：Google的机器学习框架
- DTW：动态时间规整算法
- 深度学习：CNN和LSTM网络架构

---

[中文版 README](README_CN.md) | [English README](README.md)
