# 周报：客户数据到手后的第一轮零样本评测

> **项目**：AI Adoption Clinic — Riverdale Park 镇违建照片自动分类
> **学生**：Chihao Li（技术负责人）
> **日期**：2026-04-13

---

## 我们在做什么

这周拿到了客户的第一批真实照片。按照上周定的零样本路线，本周直接拿现成的基础模型跑了评测，没有训练任何东西。除了之前已经在 proxy 数据上验证过的 CLIP，本周新加进来的是 Google 4 月刚开源的 Gemma 4 E4B-IT 多模态 VLM，4-bit MLX 量化后能在我自己的 M4 24G Mac 上本地跑。

---

## 这周拿到的数据长什么样

98 张照片，按 14 个法规代码分文件夹存放，但只有 9 个文件夹里有图，剩下 5 个是空的。每类的数量很不均衡：

| 类别 | 张数 |
|---|---:|
| Peeling Paint（剥落油漆） | 15 |
| Inoperable Vehicles（废弃车辆） | 14 |
| Long Grass / Overgrown（杂草） | 14 |
| Junk / Trash（垃圾堆积） | 13 |
| Broken Windows（破损窗户） | 12 |
| Graffiti（涂鸦） | 12 |
| Boarded Windows（钉死的窗户） | 8 |
| Damaged Roof Shingles（屋顶瓦片破损） | 7 |
| Deteriorating Chimney（烟囱破损） | 3 |

最小的类只有 3 张图。这个体量做不了监督训练，零样本是唯一选项。

整理数据时发现三件结构性的事：

**一、数据本质上是多标签的，但被存成了单标签文件夹。** 我对 98 张图算 MD5，发现有 11 张是字节级相同的副本，被巡查员同时归到了多个类别下。一张烧毁建筑的照片同时出现在 Graffiti、Long Grass、Damaged Roof Shingles 三个文件夹里。文件夹结构看起来像单标签，里面其实是被压扁的多标签。

**二、没有任何"无违建"对照照片。** 98 张全是违建，没有一张代表"巡查员去看过、判定合规"的状态。

**三、标注语义是"巡查员引用了哪条法规"，不是"画面里有什么"。** 一张破败建筑可能同时有剥落油漆、钉死的窗户、屋顶破损，但巡查员当天可能只引用了 boarded_windows，ground truth 也就只有 boarded_windows。

---

## Phase 1 vs Phase 2：同一份数据，两种问法

两个阶段都是这周做的。我先用一个直觉上最自然的框架（多分类）跑了一遍，看到一个怪现象之后改成了另一个框架（逐类二元）。98 张图没换，CLIP 和 Gemma 也没换，差别只在怎么向模型提问。

**Phase 1：多分类。** 对每张图，让模型在 9 个类别里排序，取 top-1 / top-3 当答案。
- CLIP ViT-B/32：把 9 个类别描述编码成文本向量，把图编码成图像向量，算余弦相似度，softmax 把分数压成加起来等于 100% 的概率，argmax 取最高的那一个。
- Gemma 4 E4B-IT：一个 prompt 里塞 9 个候选类别，让它返回 ranked list。

总体的 top-1 / top-3 命中率（多标签 ground truth 下）：

| 模型 | Top-1 | Top-3 |
|---|:---:|:---:|
| CLIP ViT-B/32 | 80.6% | 94.9% |
| Gemma 4 E4B-IT | 85.7% | 96.9% |

整体看上去还行，但拆到每一类就能看出问题：

| 类别 | n_pos | CLIP@1 | CLIP@3 | Gemma@1 | Gemma@3 |
|---|---:|---:|---:|---:|---:|
| boarded_windows | 10 | 100% | 100% | **20%** | 40% |
| broken_windows | 13 | 85% | 85% | 77% | 100% |
| damaged_roof_shingles | 10 | 30% | 40% | 70% | 70% |
| deteriorating_chimney | 3 | 100% | 100% | 67% | 67% |
| graffiti | 14 | 100% | 100% | 100% | 100% |
| inoperable_vehicle | 20 | 100% | 100% | 100% | 100% |
| junk_trash_accumulation | 14 | 86% | 93% | 86% | 100% |
| overgrown_vegetation | 24 | **4%** | 88% | **12%** | 88% |
| peeling_paint | 16 | **31%** | 100% | 88% | 100% |

两个奇怪的现象：

一是 Gemma 在 boarded_windows 上 top-1 召回只有 20%，但它的自由文本解释里清清楚楚写着"风化的木头、剥落的油漆、破损的瓦片"。视觉特征它都看到了，问题是在 softmax 上 9 个类别在抢同一个 100% 的概率预算，argmax 又只能挑一个赢家，Gemma 被迫把 boarded_windows 让位给一个更具体的类。

二是 CLIP 在 overgrown_vegetation 上 top-1 只有 4%，peeling_paint 只有 31%，但 top-3 都涨到了 88% / 100%。也就是说 CLIP 其实"看到"了这些类，只是 argmax 把它们压在第 2、第 3 位上而已。

考虑到数据本身是多标签的，这种压制几乎是必然发生的。

**Phase 2：逐类二元判断。** 同样的两个模型，同样的 98 张图，把"在 9 个里挑一个"改成"对每一对 (图, 类别) 独立问一个 yes/no"。
- CLIP：不再做 argmax，直接保留原始的 98×9 相似度矩阵，每一列单独看作"对这一类的独立打分"。
- Gemma：写了一个新 prompt，每次只问 "这张图是不是 X？"，强制 JSON 输出。每张图对 9 个类各跑一次，总共 882 次调用，过夜跑完。inference script 支持断点续传，因为中途 MPS 过热挂过一次。
- 然后把两者拼成 cascade：CLIP 出 top-k 候选 → Gemma 对每个候选独立判断 → 取并集作为最终预测。

---

## 结果

Phase 2 的二元问法下，Gemma 每一类的表现：

| 类别 | n_pos | AUC | Recall@0.5 | Precision@0.5 |
|---|---:|---:|---:|---:|
| deteriorating_chimney | 3 | 1.000 | 100% | 75% |
| graffiti | 14 | 0.993 | 100% | 67% |
| inoperable_vehicle | 20 | 0.977 | 90% | 72% |
| junk_trash_accumulation | 14 | 0.922 | 100% | 36% |
| overgrown_vegetation | 24 | 0.904 | 100% | 48% |
| damaged_roof_shingles | 10 | 0.850 | 70% | 50% |
| peeling_paint | 16 | 0.796 | 100% | **27%** |
| broken_windows | 13 | 0.787 | 54% | 64% |
| boarded_windows | 10 | 0.764 | **60%** | 43% |

跟 Phase 1 ranked 版直接对比：boarded_windows 的 Gemma 召回从 20% → 60%，模型、数据、训练都没动，只换了提问方式。其他几个原本被 argmax 压扁的类（broken_windows、damaged_roof_shingles）也基本拿回了召回。代价是部分类别冒出来很低的 precision —— peeling_paint 27%、junk_trash 36%、boarded_windows 43% —— 这部分后面要单独讲。

CLIP 自己也用同样的多标签口径算了逐类 AUC，可以和 Gemma binary 对比谁强谁弱：

| 类别 | CLIP AUC | Gemma AUC | 谁更强 |
|---|---:|---:|---|
| broken_windows | **0.953** | 0.787 | CLIP (+0.17) |
| boarded_windows | **0.931** | 0.764 | CLIP (+0.17) |
| peeling_paint | **0.903** | 0.796 | CLIP (+0.11) |
| overgrown_vegetation | 0.709 | **0.904** | Gemma (+0.20) |
| damaged_roof_shingles | 0.775 | **0.850** | Gemma (+0.08) |
| chimney / graffiti / vehicle / junk_trash | ~1.0 | ~1.0 | 打平 |

两个模型的强弱几乎完全错开。Gemma 在 overgrown 和 roof 这两个 CLIP 看不清的类上明显更好，CLIP 反过来在三个窗户/油漆类上明显更好。这个互补性是 cascade 之所以能跑通的根本原因。

cascade 的 F1：

| 配置 | F1 | 平均预测/张 |
|---|:---:|:---:|
| Gemma 单独二元（≡ k=9） | 0.633 | 2.43 |
| Cascade k=5 | 0.668 | 2.08 |
| Cascade k=3 | **0.703** | 1.60 |

k 越小 F1 越高，这和我的直觉相反。CLIP 对 peeling_paint、overgrown_vegetation 这些类天然就给很低的相似度。k=3 时 CLIP 直接把这些类筛掉，Gemma 没机会在它们上面乱报。cascade 真正在做的事不像"CLIP 过滤、Gemma 验证"的分级管线，更像 CLIP 和 Gemma 两个独立模型互相挡错。

两张图：第一张是上面那张 CLIP vs Gemma AUC 对比表的可视化，第二张是 CLIP 自己的逐类分数分布（橙色=正例，灰色=其他类作为留一负例），可以直观看出哪些类的正负分得开、哪些分不开。

![Per-class AUC: CLIP vs Gemma 4 binary](figures/auc_clip_vs_gemma_binary.png)

![CLIP per-class score separability](figures/clip_separability.png)

---

## 现在最大的问题：我们没有真正的负例

这是这周做完之后最重要的一件事，也是上面所有数字都要打个折看的原因。

98 张照片全是违建，没有一张是"巡查员去看过、判定合规"的对照。但要算 precision、recall、AUC，必须有正例和负例。我只能用一个 workaround：对类别 X 来说，"负例"就是没有被标 X 的那些照片。

但这些"负例"本身仍然是别的违建的正例。算 peeling_paint 的 precision 时，"负例"是被归到 graffiti、boarded_windows、broken_windows…… 的那些照片。Riverdale 的巡查照片几乎全是破败建筑，这些"负例"里大概率本来就有真实的剥落油漆，只是巡查员当天没引用 § 304.2。

这就解释了为什么 peeling_paint 在 cascade k=3 下召回 100%、精度只有 40%。Gemma 大概率不是在乱报，它是在识别画面里真实存在但 ground truth 没标注的违建，而我们的指标把它算成假阳性。

也就是说，当前的 precision 衡量的是"模型和巡查员归档习惯的一致性"，而不是"模型视觉判断的准确性"。在拿到真负例之前，所有数字都要带这个星号读。这件事不是模型问题，也不是 prompt 问题，只能靠从客户那里拿到更多/更完整的数据来解决。

---

## 下一步

**不依赖客户的事，下周自己做：**

1. 改 CLIP 在 damaged_roof_shingles 上的 prompt。这一类是 cascade Stage 1 唯一的瓶颈，CLIP 在 6/10 的真正例上根本没把它排进 top-5。换更具体的视觉描述试试。
2. 收紧 peeling_paint 的 Gemma prompt，加 "substantial / noticeable area" 这种限定词，看 FPR 会不会掉。如果掉不动，反过来更说明问题在 ground truth 不在模型。
3. 把 cascade 封装成一个能跑 demo 的模块，下次和 Ryan 同步时可以现场演示。

**要从客户那里拿的两样东西**（下次和 Ryan 同步时正式提）：

1. 大约 200 张"无违建"照片 —— 巡查员去看过、判定合规的房子。这是唯一能算出运营意义上 false positive rate 的数据。
2. 对现有 98 张做完整的多标签重标 —— 让巡查员把画面里所有可见违建都标出来，不只是当天引用的那一条。这能直接解决 peeling_paint 的精度悖论。

之前向客户提的数据请求是泛泛的"给我们更多标注数据"。这次的请求是具体的：要负例做 FPR、要补全标签做 precision。这种具体的请求更容易过 Town Hall 的合规审查。

**如果客户那边走不通的兜底方案**：

Town Hall 的合规审查一直是数据这块的瓶颈，Ryan 那边能不能拿到这两批数据我没有把握。如果走不通，有两个备份方向：

1. **自己构造负例池。** 用我们之前已经收集的 proxy 数据集（街景、Google Open Images 里的住宅外观），人工挑出明显合规的房屋照片，凑出一批"非违建"对照。这种负例不会有 Riverdale 巡查照片的真实分布，但至少能把 false positive rate 算出一个数量级，比当前完全没有要好。
2. **对现有 98 张做多标签重标。** 巡查员的标注是"引用了哪条法规"，这个改不了，但画面里实际有什么违建是可以看着照片重新标的。让 9 个类别在标注层面互为正负 —— 一张被巡查员归到 boarded_windows 的照片，如果画面里同时有剥落油漆，就同时打上 peeling_paint 标签。重标后的 ground truth 拿来重算 precision，能直接验证"peeling_paint 27% 是模型乱报还是 ground truth 漏标"这件事。

这两个方向都不依赖外部数据，技术上都能走通。具体怎么推进、谁来做，等下次和 Ryan 同步、以及和团队对齐之后再定。

---

## Reproducibility

所有代码、脚本和报告源文件都在 `https://github.com/psylch/inst623-riverdale-code-enforcement`。从干净环境复现：

```bash
# Clone the repository and enter the project root
git clone https://github.com/psylch/inst623-riverdale-code-enforcement.git
cd inst623-riverdale-code-enforcement

# Install Python dependencies (uses uv for Python 3.12)
uv sync

# Place client photos under data/client-data/ before running
# (one subdirectory per violation code, matching the folder names in the
#  official Riverdale Park taxonomy). The client dataset itself is not
#  redistributed in this repository for privacy reasons.

# CLIP separability + top-k analysis (~2 minutes)
uv run python scripts/run_clip_separability.py

# Gemma binary verification (~47 minutes, resumable)
uv run python scripts/run_gemma_binary.py
# Monitor progress in another terminal:
tail -f checkpoints/client_gemma4_binary_stream.jsonl

# Cascade evaluation and figures
uv run python scripts/analyze_binary_results.py
```

所有路径相对于仓库根目录。所有实验产物（预测结果、相似度矩阵、审计日志、图表）都落在 `checkpoints/` 和 `reports/figures/` 下。整条流水线在 24 GB 统一内存的 Apple Silicon Mac 上可以端到端跑通，模型权重首次运行时从 Hugging Face 下载并本地缓存，不需要任何外部算力。
