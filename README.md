# FootballerAnalyzer

基于摄像头的足球数据分析平台。

**当前阶段**：调研 ✅ → 需求 ✅ → **选型 ✅** → 设计 ⬜ → 开发 ⬜

## 📌 文档入口

**[docs/README.md](docs/README.md)** —— 文档索引：每份文档的价值、主要内容、阅读路径与维护约定。

## 三条最重要的调研结论（TL;DR）

1. **参考项目 `abdullahtarek/football_analysis` 只能当读物**：单应性 4 点硬编码、fps 写死 24、球不跟踪、无身份保持、**无 LICENSE**。流水线概念对，实现全要重写。
2. **没有任何开源项目能直接变成平台**：现成的只有零件（MIT 的 `roboflow/sports` 标定+分队、`torchreid` 特征、`sn-trackeval` 评测）；**事件层、身份层、数据层、服务层是公开空白**，必须自研。
3. **本机（无独显）够用**：整场 90 分钟 1080p 逐帧约 3.8 小时、抽帧约 1.1 小时；检测走 ONNX Runtime **DirectML（核显 Radeon 780M）比 CPU 快 2.3×**。实时不行，离线没问题。

## 目录结构

```
FootballerAnalyzer/
├── docs/
│   ├── README.md                            # ★ 文档索引（从这里进）
│   ├── 00-总报告-调研结论.md                 # ★ 主文档：能不能做、坑在哪
│   ├── 01-参考仓库清单与许可.md              # 23 个仓库的许可与可用度评级
│   ├── 02-本机实测数据.md                    # 本机基准测试与结论
│   ├── 03-核心源码审计与可复用清单.md         # 逐项目审计：哪段能抄、哪段必坏（带行号）
│   ├── 04-主干选型清单.md                    # ✅ 已定稿：D0–D13 十四个决策点（效率优先）
│   ├── 05-需求确认单.md                      # 已填写：6 条必答 + 5 条选答
│   ├── 06-需求回执与范围确认.md               # 需求解读 + 机位实测硬约束 + 范围砍法
│   ├── 10-技术调研-SOTA与数据集.md           # 学术 SOTA / 数据集 / 关键论文
│   ├── 11-从轨迹到统计指标.md                # 跑动/速度/控球/传球的算法与坑
│   ├── 12-硬件与部署调研.md                  # 摄像机选型 / 标定误差 / 造价
│   └── bench/                               # 基准测试原始 JSON
├── scripts/
│   ├── clone_refs.ps1                       # 浅克隆全部参考仓库到 third_party/
│   ├── setup_bench_env.ps1                  # 一键建立 .venv 基准环境
│   ├── bench_cpu.py                         # 检测/光流/ONNX-DirectML 基准
│   └── bench_pipeline.py                    # 端到端吞吐基准（含 ByteTrack）
├── third_party/                             # 23 个参考仓库源码（★ 已 gitignore，不入库）
├── models/                                  # 模型权重（★ 已 gitignore）
└── .venv/                                   # 本地 Python 环境（★ 已 gitignore）
```

## 快速开始

```powershell
# 1) 建立基准环境（约 200 MB 下载：torch CPU + ultralytics + onnxruntime-directml）
& .\scripts\setup_bench_env.ps1

# 2) 复现本机基准数据
& .\.venv\Scripts\python.exe .\scripts\bench_cpu.py --frames 20
& .\.venv\Scripts\python.exe .\scripts\bench_pipeline.py --frames 300 --imgsz 640

# 3) 需要时重新克隆参考仓库（已存在会跳过）
& .\scripts\clone_refs.ps1
```

## 关于第三方源码与素材

`third_party/` 下是 23 个浅克隆的调研用仓库（约 1.9 GB），**已在 `.gitignore` 中排除，不会上传**。
各仓库版权与许可归原作者所有，本目录仅用于本地阅读与对照，许可条款参见 [docs/01](docs/01-参考仓库清单与许可.md) —— 其中 `ultralytics` 与 `boxmot` 是 AGPL-3.0，`sn-gamestate`/`sportslabkit` 是 GPL-3.0，商用前务必确认。

`.gitignore` 同时排除了：模型权重（`*.pt/*.onnx/...`）、视频与数据集、虚拟环境、密钥，以及**含第三方版权画面的参考截图**（`docs/image.png`、`docs/bench/*.png` —— 本仓库为 public，这些图仅本地保留）。

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 调研 | ✅ 完成 | [00](docs/00-总报告-调研结论.md) 总报告 + [01](docs/01-参考仓库清单与许可.md)/[02](docs/02-本机实测数据.md)/[03](docs/03-核心源码审计与可复用清单.md) + 专题 [10](docs/10-技术调研-SOTA与数据集.md)/[11](docs/11-从轨迹到统计指标.md)/[12](docs/12-硬件与部署调研.md) |
| 需求 | ✅ 完成 | [05 需求确认单](docs/05-需求确认单.md)（已填写）、[06 需求回执与范围确认](docs/06-需求回执与范围确认.md) |
| 选型 | ✅ 完成 | [04 主干选型决策](docs/04-主干选型清单.md)：D0–D13 已定稿（效率优先原则）；测试素材已就位 |
| 设计 | ⬜ 未开始 | 架构、数据模型、接口定义、开发排期**尚未产出** |
| 开发 | ⬜ 未开始 | 素材已就位（`data/clips/`，本地不入库）；下一步可开工骨架 |
