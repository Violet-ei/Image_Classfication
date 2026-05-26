# 四种图像分类方法评估系统（删除 DINOv2 版）

本项目是在上一版“五种图像分类方法演示系统”基础上的修改版。按照新的需求：

1. 已删除 DINOv2，避免 `torch.hub` / GitHub 权重授权问题；
2. 支持上传有标注数据集，统计各个模型的分类准确率；
3. 自动比较各模型分歧最大的样本，并给出原因分析。

当前支持 4 种方法：

| 方法 | 默认实现 | 说明 |
|---|---|---|
| DenseNet | `torchvision.models.densenet161` | 最大常用 DenseNet 配置之一，密集连接与特征复用 |
| ConvNeXt | `timm/convnext_xlarge.fb_in22k_ft_in1k_384` | 现代化卷积网络，22K 预训练后 1K 微调，高分辨率 XLarge 权重 |
| ConvNeXt V2 | `timm/convnextv2_large.fcmae_ft_in22k_in1k_384` | ConvNeXt + FCMAE 自监督预训练，高分辨率 Large 权重 |
| EVA | `timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 大规模 MIM 视觉表征模型，Large 权重 |

---

## 1. 环境安装

推荐 Python 3.9 或以上。

```bash
pip install -r requirements.txt
```

如果已经装过 PyTorch，可先注释 `requirements.txt` 中的 `torch` 和 `torchvision`，再安装其余依赖。

---

## 2. 启动界面

```bash
python app.py
```

启动后浏览器打开：

```text
http://127.0.0.1:7860
```

界面包含两个标签页：

1. **单张图片分类**：上传一张图片，展示各模型 Top-K 分类结果；
2. **数据集准确率与分歧分析**：上传有标注数据集 zip，统计各模型准确率并分析分歧样本。

---

## 3. 数据集格式要求

数据集需要采用 **class-folder** 结构，即每个类别一个文件夹：

```text
dataset/
├── airplane/
│   ├── airplane_001.jpg
│   └── airplane_002.jpg
├── bird/
├── car/
├── cat/
├── dog/
├── flower/
└── ship/
```

然后压缩为 `dataset.zip` 上传即可。

---

## 4. 准确率统计方式

由于这些模型使用的是 ImageNet-1K 分类头，输出是 ImageNet 的 1000 个细粒度类别，而你的数据集标签通常是粗类别，例如：

- `dog`
- `cat`
- `car`
- `ship`
- `airplane`

因此系统会先把 ImageNet 细粒度预测映射回粗类别，再计算准确率。例如：

- `Labrador retriever`、`golden retriever`、`beagle` → `dog`
- `tabby`、`Persian cat`、`Siamese cat` → `cat`
- `sports car`、`cab`、`limousine` → `car`
- `airliner`、`warplane` → `airplane`
- `container ship`、`liner`、`speedboat` → `ship`

系统输出两类指标：

| 指标 | 含义 |
|---|---|
| Top-1 coarse accuracy | 模型 Top-1 预测映射到粗类别后是否等于真实类别 |
| Top-K contains true | Top-K 预测中是否至少有一个能映射到真实类别 |

如果你使用其他类别，可以修改 `label_mapping.py` 中的 `COARSE_KEYWORDS` 和类别别名。

---

## 5. 模型分歧分析

系统会统计每张图片在不同模型下的预测结果，并按照以下因素找出分歧最大的样本：

- 不同模型预测出的粗类别数量；
- 预测投票分布是否分散；
- 模型之间置信度差异；
- 有多少模型预测正确。

分歧样本表会给出：

- 图片文件名；
- 真实类别；
- 各模型预测结果；
- 投票分布；
- 分歧分数；
- 自动原因分析。

这些原因分析是基于模型输出的启发式判断，适合写进课程实验报告，例如：

> 对于复杂背景或多目标图片，不同模型可能关注不同区域，导致预测结果分散；对于主体较小或类别边界不清晰的图片，模型置信度通常较低，且更容易受到背景纹理干扰。

---

## 6. 命令行评估

除了界面，也可以直接用命令行评估：

```bash
python evaluate_dataset.py --dataset dataset.zip --models all --topk 5 --batch-size 8
```

输出文件默认保存在 `eval_outputs/`：

```text
eval_outputs/
├── model_accuracy_summary.csv
├── all_model_predictions.csv
└── largest_disagreement_samples.csv
```

---

## 7. 项目结构

```text
image_classification_4models_dataset_eval/
├── app.py                  # Gradio 软件界面
├── classifier.py           # 模型加载、单图推理、数据集评估、分歧分析
├── data_utils.py           # 数据集解压与 class-folder 读取
├── label_mapping.py        # ImageNet 细粒度标签到数据集粗类别的映射
├── evaluate_dataset.py     # 命令行评估脚本
├── requirements.txt
├── run_windows.bat
└── run_linux_mac.sh
```

---

## 8. 报告中可以这样说明

本系统实现了 DenseNet、ConvNeXt、ConvNeXt V2 和 EVA 四种图像分类方法，并设计了可视化软件界面展示分类结果。系统不仅支持单张图像的 Top-K 分类结果展示，还支持上传有标注数据集，自动统计各模型在该数据集上的粗类别分类准确率。由于所采用的公开预训练模型主要基于 ImageNet-1K 分类标签，系统进一步设计了细粒度 ImageNet 类别到数据集粗类别的映射规则，从而能够更合理地评估模型在自定义数据集上的表现。同时，系统比较不同模型在同一样本上的预测差异，筛选出分歧最大的样本，并从多目标、复杂背景、主体不明显、类别边界模糊等角度给出原因分析。
