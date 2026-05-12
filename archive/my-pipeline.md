# 我的技术 Pipeline

> 个人执行文档，记录对齐后的技术方案和决策。

---

## 背景

- 客户（Riverdale Park）有 ~1000 张违规照片，600-700 种 violation code，我们聚焦 top 20-30 类
- 客户数据大概率会拖，老师建议技术先行
- 目标：Inspector 拍照 → 系统返回 top-N 可能的违规类型 + 置信度 → Inspector 确认/拒绝

## 关键决策

### 不需要 YOLO

之前做植物识别是 YOLO 检测 + 分类两阶段。这次不需要：
- Inspector 拍的照片本身就是一个违规场景，不需要从大场景里框目标
- 直接整图分类，少一个阶段 = 少一半工作量
- 如果后续扩展到车载摄像头街景巡检（像 CityDetect 那样），再加 YOLO 检测阶段

### 先 multi-class，不急 multi-label

理论上一张图可能多个违规，但实际上 inspector 拍照有针对性，一张照片对应一个主要问题。先 softmax 跑通，后面有需要再切 sigmoid + BCE。

### 本地训练可行

M4 24GB 完全够用：
- Linear probe：提 features 几分钟，训 head 秒级
- LP-FT 全量微调：batch 16，1000 张图，50 epochs 半小时内
- `device = "mps"`

---

## Pipeline 四步走

### Step 1: CLIP Zero-Shot Baseline

**目的**：零训练成本拿一个 baseline，同时验证"用语义描述做分类"这条路走不走得通。

**做法**：
- 模型：`openai/clip-vit-base-patch16`
- 把 20-30 个违规类别写成自然语言 prompt（e.g. "a photo of overgrown grass and vegetation"）
- 图片过 CLIP → 和每个 prompt 算相似度 → 排序输出 top-3
- 用 proxy 数据集（BD3 + TACO）验证

**预期**：top-1 准确率 40-60%，top-3 可能 70%+。不高但够当 baseline。

**核心价值**：客户数据没到就能 demo，"不需要训练就能工作" 对 client 更有说服力。

### Step 2: 准备 Proxy 数据集

**目的**：客户数据等不来，先用公开数据集搭 pipeline。

**数据源**：
| 数据集 | 数量 | 覆盖的违规类型 |
|--------|------|---------------|
| BD3 | 3,965 张 | 建筑缺陷（裂缝、剥落、霉斑） |
| TACO | 1,500 张 | 垃圾/杂物堆积 |
| Grass-Weeds | 2,486 张 | 植被过度生长 |
| Aerial Dumping | 1,555 张 | 非法倾倒 |

**标签映射**：
```
BD3 crack类       → structural_damage
BD3 peeling/spalling → exterior_deterioration
TACO 全部          → trash_debris
Grass-Weeds        → overgrown_vegetation
Aerial Dumping     → illegal_dumping
```

统一成 5 个大类先跑，等真实数据来了再细分到 20-30 类。

### Step 3: 模型对比实验

**三个模型都跑一遍**：

| 模型 | 思路 | 预期表现 |
|------|------|---------|
| **DINOv2 ViT-B/14** | 自监督预训练，frozen features 就很强 | 最高（特征质量好）|
| **EfficientNetV2-S** | ImageNet 预训练 fine-tune，我熟悉的路线 | 稳定，保底方案 |
| **CLIP zero-shot** | Step 1 的结果直接拿来对比 | 最低，但零成本 |

**训练策略统一用 LP-FT**：
```
Phase 1: Linear Probe
  - Freeze backbone
  - 只训分类头，20-50 epochs，lr=0.001，SGD

Phase 2: Full Fine-Tune（从 Phase 1 初始化）
  - 全部解冻
  - lr=1e-5~5e-5（backbone）/ 1e-4（head）
  - AdamW, weight_decay=0.01
  - Cosine annealing + 5% warmup
  - Early stopping on val F1
```

**数据增强**：
- 基础：RandomResizedCrop, HorizontalFlip, BrightnessContrast, ColorJitter
- 天气模拟：RandomRain, RandomFog（inspector 什么天气都拍）
- 正则化：CoarseDropout, MixUp, CutMix
- 库：Albumentations + timm 内置

**评估指标**：
- Per-class precision / recall / F1
- Top-3 accuracy（正确标签在前 3 预测里吗）
- Macro F1（类别不均衡下的公平指标）
- Confusion matrix 找哪些类容易混淆

### Step 4: 切换真实数据

客户数据到了之后：
1. 拿到 Riverdale Park 的 violation code taxonomy，和 Niping 一起做 code → category 映射
2. EDA：看类别分布、图片质量、有没有标注问题
3. 如果类别和 proxy 有重叠 → 合并数据一起训；没重叠 → 只用真实数据
4. 重跑 LP-FT pipeline，指标对比
5. Niping 跑 fairness analysis（按 neighborhood 分层评估 error rate）

---

## 工具栈

| 工具 | 用途 |
|------|------|
| PyTorch + MPS | 训练框架，M4 本地跑 |
| timm | 模型库（DINOv2, EfficientNetV2, ConvNeXt 都有） |
| HuggingFace Transformers | CLIP 模型 |
| Albumentations | 数据增强 |
| scikit-learn | 评估指标 |
| Jupyter notebook | 实验记录，最终交付物也是 notebook |

---

## 行业参考

| 公司 | 做法 | 关键数据 |
|------|------|---------|
| CityDetect (PASS AI) | 车载摄像头 + AI 自动巡检 | Dallas 试点 95% 准确率，3000+ 违规/4天，$13M 融资 |
| Forerunner | 治理优先，自动生成引用具体法条的违规通知 | Human-in-the-loop |
| DataGrid | AI agent 处理违规投诉和案件管理 | 自动化流程 |

CityDetect 的隐私做法（auto-blur 人脸/车牌）可以作为我们 responsible AI 部分的参考。

---

## 时间线

| 时间 | 做什么 |
|------|--------|
| 本周 | CLIP zero-shot baseline notebook |
| 第 2 周 | 下载 proxy 数据集 + EDA + DINOv2 linear probe |
| 第 3 周 | 模型对比实验（DINOv2 vs EfficientNetV2 vs CLIP）|
| 数据到了 | 换真实数据重跑，fairness analysis |
| 最后 2 周 | 整理 report + prototype demo + presentation |
