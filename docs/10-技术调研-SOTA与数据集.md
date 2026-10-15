# 足球视频/摄像头分析：学术 SOTA 与数据基准（2023–2026，含奠基工作）

> 所有数字均取自论文/官方页面原文；未直接读到原文的标「未核实」。

## ① 结论速览

1. **足球 MOT 仍是 tracking-by-detection 天下**：检测器（YOLOX / YOLOv8-v11 / RF-DETR）+ 关联器（ByteTrack / BoT-SORT / OC-SORT / Deep-EIoU）。SportsMOT 论文自证：体育场景里 Kalman 线性运动假设会翻车（ByteTrack 去掉运动预测反而 +7.4 HOTA）。
2. **SportsMOT test HOTA**（720p/25fps，240 clips，3 运动）：ByteTrack 62.8（Train+Val 64.1）、BoT-SORT 68.7、OC-SORT 71.9（Train+Val 73.7）、MixSort-OC 74.1、Deep-EIoU 75.3（arXiv v1；官方仓库/后续版本标称 77.2）。
3. **SoccerNet tracking 2023 冠军 Kalisteo 75.61 HOTA**（YOLOX + TrackMerger v2 + MGN ReID 迭代合并 tracklet），基线仅 42.38 → 瓶颈在关联不在检测。
4. **固定鱼眼全景是另一条赛道**：SoccerTrack Challenge 2025 冠军 GTATrack（Deep-EIoU + Global Tracklet Association）HOTA 0.60。
5. **FieldMOT（CVPRW'25）提「register-then-track」**：先标定回球场坐标再跟踪，专治转播多机位切换。
6. **身份保持两极分化**：跨机位 ReID 近乎饱和（SoccerNet ReID 2023 冠军 CLIP-ReIdent 93.26 mAP，基线 91.68）；真卡点是**号码识别**（2023 冠军 92.85% tracklet acc）与**长时 tracklet 合并**（GTA / split-and-merge）。
7. **球队分类**在 GSR 里普遍退化为「球衣颜色聚类 + 位置先验」，工程上已相当可靠。
8. **标定**：TVCalib 把 homography 重构成可微相机标定；NBJW/PnLCalib 用几何层级关键点 + DLT/优化；SoccerNet 2023 calibration 冠军 Sportlight Score 0.5535（Acc@5 0.7322）。ProCC 论文警告：WC14/SoccerNet 的 homography 指标会系统性偏袒 homography 方法。
9. **GSR 是最接近产品化的任务**：GS-HOTA = LocSim×IdSim；基线 23.36→29.01，2024 冠军 63.81，2025 冠军 63.90 —— 两年几乎原地踏步。参赛方案高度趋同：关键点标定 + Deep-EIoU/BoT-SORT + GTA + VLM/CLIP 认号码。
10. **球**：小目标 + 运动模糊 + 遮挡三难；出界/最后触球目前主要靠半自动标注或球内传感器（FIFA），纯视觉尚无公认基准。

## ② 分类详述

### 1. 体育 MOT
- ByteTrack / OC-SORT / BoT-SORT / StrongSORT：行人域经典，迁移到体育时运动模型失配（见 SportsMOT Tab.6 / Deep-EIoU Tab.3）。基线数字见 [SportsMOT](https://arxiv.org/abs/2304.05170)、[Deep-EIoU](https://arxiv.org/abs/2306.13074)。
- **Deep-EIoU**（WACV'24）：抛弃 Kalman，用迭代放大的 ExpansionIoU 做运动无关关联 + OSNet 深度特征；SportsMOT 75.3 HOTA（[代码](https://github.com/hsiangwei0903/Deep-EIoU)）。
- **GTATrack**（SoccerTrack 2025 冠军）：Deep-EIoU + 全局 tracklet 关联两阶段，治鱼眼畸变 + 尺度剧变，HOTA 0.60、FP 982（[arXiv](https://arxiv.org/abs/2602.00484)）。
- **FieldMOT**（CVPRW'25）：register-then-track，用开放词表基础模型取球场 anchor/keypoint，合成足球数据集上验证（[论文](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Chen_FieldMOT_A_Field-Registered_Multi-Object_Tracking_for_Sports_Videos_CVPRW_2025_paper.html)）。
- **SoccerNet 2023 tracking** 冠军 Kalisteo：YOLOX + 两次匈牙利匹配 + Kalman（带相机运动补偿）+ tracklet 分裂/合并（[报告](https://arxiv.org/abs/2309.06006)）。
- 相关基准：[TeamTrack](https://arxiv.org/abs/2404.13868)（全景多运动）、[TrackID3x3](https://arxiv.org/abs/2503.18282)（固定机位 Track-ID 任务）。

### 2. 身份保持 / 号码 / 球队
- **ReID**：SoccerNet ReID 冠军 **CLIP-ReIdent**（用 CLIP ViT-L14 在 ReID 数据集上零样本打伪标再微调，93.26 mAP / 基线 91.68）。2023 排名表在 [SoccerNet 2023 报告](https://arxiv.org/abs/2309.06006)。
- **号码识别**：主流三阶段「文本框检测（DBNet++/MMOCR/YOLO）→ OCR（PP-OCRv3/PaddleOCR）→ tracklet 多数投票」，2023 冠军 92.85%；[sn-jersey](https://github.com/SoccerNet/sn-jersey)；2025 GSR 里改用 VLM（LLaMA-3.2-Vision / Qwen2-VL + function calling）与 CLIP+PARSeq。
- **tracklet 关联**：`GTA`（Global Tracklet Association）在 2024/2025 GSR 前几名里几乎人手一个；冠军用 split-and-merge（按身份预测一致性切、按 ReID 相似度合）。
- **球队分类**：jersey/uniform 颜色聚类 + x 坐标先验（冠军方案），或用 Siamese embedding + KMeans + RandomForest 全局聚类（2024 参赛队）。

### 3. 相机标定 / 单应性
- **TVCalib**（WACV'23）：把球场注册当相机标定，可微目标函数从线段对应学位姿+焦距，一步到位（[arXiv](https://arxiv.org/abs/2207.11709)，[代码](https://github.com/MM4SPA/tvcalib)）。
- **NBJW → PnLCalib**（CVPRW'24 → CVIU'26）：几何层级关键点生成 + 线交点增广 + DLT 极简标定，后扩展为 points-and-lines 非线性优化；支持单目 3D 标定与多视角（[arXiv:2404.08401](https://arxiv.org/abs/2404.08401)，[代码](https://github.com/mguti97/No-Bells-Just-Whistles)）。NBJW 是 GSR 参赛队事实标准。
- **Sportlight**（SoccerNet 2023 calibration 冠军）：HRNetV2-w48 检 57 关键点 + 23 条线，OpenCV 标定 + RMSE 投票，Score 0.5535 / Acc@5 0.7322 / Completeness 0.7559（[论文](https://arxiv.org/abs/2410.07401)）。同榜基线 Score 0.0833 —— 注意该 metric = Acc@5 × Completeness，Completeness 高的方案（Spiideo 0.9997）Acc 只有 0.5295。
- **ProCC**（CVPRW'24）：提出模型无关的 JaC_τ 指标，指出依赖 GT homography 的老指标无法评价鱼眼畸变与非平面元素（[arXiv](https://arxiv.org/abs/2404.09807)）。
- 适用条件：SoccerNet 数据集 = 转播多机位（21,132 图/2022 版，25,506 图 + 226,305 polylines/2023 版，960×540）；单目固定机位标定通常更简单（畸变可离线标定），但鱼眼需显式径向畸变模型。

### 4. Game State Reconstruction
- **任务**：单目转播 → 2D 俯视小地图上的全员位置 + 角色 + 球队 + 号码。
- **数据集 SoccerNet-GSR**：200 段 30s 视频，9.37M 球场线点 + 2.36M 球员场上坐标（带 role/team/jersey）[arXiv](https://arxiv.org/abs/2404.11335) / [代码](https://github.com/SoccerNet/sn-gamestate)。
- **指标 GS-HOTA**：LocSim（真实 2D 距离）× IdSim（号码/角色/球队），替代 HOTA 的图像域 IoU 匹配。
- **性能**：基线 23.36（2024 报告）→ 更新基线 29.01；2024 冠军 **Constructor.Tech 63.81**（YOLOv5 + CNN-Transformer 相机参数 + 71 关键点精修 + pitch 坐标 DeepSORT + 5 角色分类）；2025 冠军 **KIST-GSR 63.90**（YOLO-X + Deep-EIoU + OSNet + LLAMA-3.2-Vision 多模态自回归出号码/角色，GS-DetA 51.36 / GS-AssA 79.55），第 2 名 Constructor.Tech 63.81。（[2024 报告](https://arxiv.org/abs/2409.10587)、[2025 报告](https://arxiv.org/abs/2508.19182)）
- 另一条路线：[From Broadcast to Minimap](https://arxiv.org/abs/2504.06357)（YOLOv5m + SegFormer 相机参数 + DeepSORT-ReID + 朝向预测），称 SOTA。

### 5. 球检测/轨迹与出界判定
- 难点：球在转播画面里常只有 5–15 px、高速运动模糊、被球员/球门遮挡。
- **TOTNet**（2025）：遮挡感知时序跟踪做鲁棒球检测（[arXiv](https://arxiv.org/abs/2508.09650)）；早期代表 [FootAndBall](https://github.com/jac99/FootAndBall)（球员+球联合检测）。
- 球事件：SoccerNet Ball Action Spotting（2024 冠军 mAP@1 从基线 56.15 提到 73.39；2025 扩展为 Team Ball Action Spotting）。
- 出界/最后触球：纯视觉尚无公开标准基准，工程上见 [Semi-automated last touch detection for out-of-bounds possession decisions in football](https://link.springer.com/article/10.1007/s12283-025-00518-3)（半自动）；FIFA 的 Out-of-bounds technology 走球内传感器路线。

### 6. 单目 vs 多摄
- 单目转播是全部主基准，但**画面外球员不可见**，拿不到完整阵型与压迫数据。
- 多机位/多视角融合：[Graph-Based Multi-Camera Soccer Player Tracker](https://arxiv.org/abs/2211.02125)（直接在多相机原始检测热图上跟踪，规避标定误差）；[SoccerNet-v3D](https://arxiv.org/abs/2504.10106)（用转播回放做 3D 场景理解）；[WorldPose](https://arxiv.org/abs/2501.02771)（世界杯多视角 3D 人体姿态）；SoccerNet 2024/2025 的 Multi-View Foul Recognition 任务（多路同步机位分类犯规类型与严重度）。
- 自建系统可行性：**SoccerTrack v2** 是最强证据 —— 10 场大学比赛、932 分钟 4K 固定机位全景、逐帧米制坐标 + 号码 + 身份 + 角色 + 球队 + 12 类球事件，并报告 GS-HOTA 随序列长度上升而退化（[arXiv](https://arxiv.org/abs/2508.01802)，[项目页](https://atomscott.github.io/SoccerTrack-v2/)）。结论：固定机位/鱼眼自建系统**在标定与身份一致性上更容易**（无镜头切换），代价是畸变与远景目标极小。

## ③ 数据集表

| 名称 | 内容 | 规模 | 许可 | 获取方式 |
|---|---|---|---|---|
| SoccerNet v1/v2 | 完整转播比赛 + 事件标注 | 500 场，17 类 action（标注总量未核实） | 研究用途；含版权视频需接受 NDA | [HF 组织](https://huggingface.co/SoccerNet)（受保护视频 gated 申请）、[pip SoccerNet](https://pypi.org/project/SoccerNet/) |
| SoccerNet-v3 | 空间标注：球员/球/球门/线框 + pitch localization + calibration + ReID + jersey | 标定 25,506 图 / 226,305 polylines（2023 版）；ReID 340,993 thumbnails、400 场、6 联赛；jersey 2,853 tracklets | 同 SoccerNet NDA | 同上 |
| SoccerNet-v4 | 未核实到官方 v4 论文；各任务数据现统一由 HF 组织发布 | 未核实 | 同 SoccerNet | [huggingface.co/SoccerNet](https://huggingface.co/SoccerNet) |
| SoccerNet-GSR | 30s 序列 + 球场线点 + 球员场上坐标（role/team/jersey） | 200 段，9.37M 线点，2.36M 球员位置 | 代码/标注开源（视频部分可能仍受 NDA） | [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) |
| SoccerNet-Tracking | 主摄像机 30s 片段 tracklet（球员/裁判/球），含号码与球队 | 200 段（未核实总帧数） | 同 SoccerNet | [sn-tracking](https://github.com/SoccerNet/sn-tracking) |
| SportsMOT | 篮球/足球/排球 MOT，MOT17 格式 | 240 clips、150K+ 帧、1.6M+ bbox、1280×720@25fps | **CC BY-NC 4.0** + 竞赛条款（禁止再分发） | [CodaLab 注册](https://codalab.lisn.upsaclay.fr/competitions/12424)、[GitHub](https://github.com/MCG-NJU/SportsMOT) |
| TeamTrack | 全景多运动 MOT（足球/篮球/手球） | 未核实 | 未核实 | [项目页](https://atomscott.github.io/TeamTrack/) |
| SoccerTrack v2 | 固定机位 4K 全景，米制坐标/号码/身份/角色/球队 + 12 类球事件，带 match-level split | 10 场、932 分钟 | 论文声明开放（具体 license 未核实） | [项目页](https://atomscott.github.io/SoccerTrack-v2/) |
| DeepSportradar v1/v2 | 篮球/足球实例分割等 | v1 2022、分割挑战 v2 2023（图像量未核实） | 未核实（学术竞赛用） | [GitHub](https://github.com/DeepSportradar/instance-segmentation-challenge) |
| ISSIA-CNR | 单固定机位手标球员/球，MOT 格式 | 6 段序列、320×240（总帧数未核实） | 学术免费 | 需向 CNR 索取（入口未核实） |
| StatsBomb open-data | 事件数据（含 360 freeze-frame） | 免费开放子集（场次未核实） | 免费，要求署名 StatsBomb | [hudl/open-data](https://github.com/hudl/open-data) |
| Metrica Sports sample-data | 匿名化 tracking + event | 3 场 | 未核实 | [GitHub](https://github.com/metrica-sports/sample-data) |

> 「MetricSports」未检索到对应足球数据集，疑为 **Metrica Sports** 之误（未核实）。

## ④ 对你选题最关键的 5 篇

1. [SoccerNet Game State Reconstruction](https://arxiv.org/abs/2404.11335) — GSR 任务定义 + SoccerNet-GSR + GS-HOTA，一切的地基。
2. [SoccerNet 2025 Challenges Results](https://arxiv.org/abs/2508.19182) — 当期四个任务全部冠军方案的模块清单与 GS-HOTA 完整榜单（做竞品分析最高效）。
3. [SportsMOT](https://arxiv.org/abs/2304.05170) + [Deep-EIoU](https://arxiv.org/abs/2306.13074) + [GTATrack](https://arxiv.org/abs/2602.00484) — 体育 MOT 的基线、去 Kalman 路线与全景鱼眼最新冠军。
4. [No Bells, Just Whistles / PnLCalib](https://arxiv.org/abs/2404.08401) — 单目 3D 标定的工程最优解，配套 [代码](https://github.com/mguti97/No-Bells-Just-Whistles) 可直接用。
5. [SoccerTrack v2](https://arxiv.org/abs/2508.01802) — 固定机位自建系统的数据规格与已知失效模式（GS-HOTA 随序列变长退化），直接对应你的「手机/鱼眼自建」可行性判断。
