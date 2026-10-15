# FootballerAnalyzer

基于固定机位视频的足球球员追踪、场地投影与跑动分析原型。

**当前阶段**：调研、需求、选型与设计已完成；最小闭环原型已实现。实片追踪与前端对拍已验证，距离精度仍待独立参考路程验证。

## 现在可以做什么

- YOLOv8n 人物检测、ByteTrack 跟踪、球衣颜色分组，缓存原始像素轨迹。
- 浏览器同帧展示原图检测框、脚点与球场投影；支持播放、逐帧、源帧跳转和联动选择。
- 点选场地对应点和可靠区域，检查白线叠加；保存草稿或确认标定。
- 建立持久球员档案，按源帧区间绑定、拆分或排除轨迹。
- 重算距离、速度、有效时间和热图，导出 HTML、叠加视频与 Parquet；查询选定报告中的球员历史。

## 快速开始（Windows / PowerShell）

在项目根目录执行，推荐 Python 3.12；本机已建好的 `.venv` 可直接复用。

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -e . -c configs/constraints-tested.txt
```

已安装本项目时，可直接从这里开始：

```powershell
# 视频与模型为本地文件，首次在新机器上使用时需要自行准备
# 默认：data/clips/goodview_static_5s.mp4 和 models/yolov8n.pt
& .\.venv\Scripts\python.exe -m footballer_analyzer track --config configs/demo.yaml

# 跟踪成功后，选择刚生成的运行目录
$runDir = (Get-ChildItem .\outputs -Directory -Filter 'run_*' | Sort-Object Name -Descending | Select-Object -First 1).FullName

# 只适用于同名演示素材；未确认的粗选点不能作为精度依据
& .\.venv\Scripts\python.exe -m footballer_analyzer calibrate --run $runDir --points configs/calibration_demo.yaml

& .\.venv\Scripts\python.exe -m footballer_analyzer serve --port 8765
```

打开 **[http://127.0.0.1:8765/](http://127.0.0.1:8765/)**。服务按 `Ctrl+C` 停止；换用其他输出目录时加 `serve --output-root <目录>`。视频、模型、球员档案与分析产物保存在本机，未纳入 Git。

### 前端操作顺序

1. 左侧选择运行；点击列表、图像中的球员框或球场圆点，联动核对相同轨迹 ID。
2. 用上一帧、下一帧、滑条或输入源帧号后点击“跳转”定位。预览最高 10 FPS；源帧号与原始时间保持一致。
3. 在“场地标定”填写实际尺寸，在同一参考帧点选至少 4 对非共线点（建议 6–12 对），填写对应米制坐标；按顺序点选可靠地面区域。
4. 保存草稿，检查首、中、末帧白线叠加、尺寸与可靠区域。切镜或机位漂移应缩短窗口并重新分析。确认检查后再勾选标定确认；草稿允许位置预览，但阻止距离报告。
5. 在“身份与轨迹”建立球员档案并绑定区间。同一人断轨后把新 ID 绑定到同一档案；同 ID 换人时分别绑定不同区间；裁判或场外人员可以排除。编辑已有绑定时先移除原条目，再保存替代区间。
6. 在“跑动统计”点击“生成 / 重算报告”；点击统计行查看速度与热图，打开完整报告下载叠加视频和数据。修改绑定或标定后会提示旧报告过期，重算保留历史版本。
7. “导出对拍图”生成当前帧的画面与球场并排 PNG，再点“保存 PNG”。

`track_id` 只在一次运行内有效，`player_id` 才是跨场身份；`home/away` 为人工确认的本场队伍，颜色簇不自动代表真实主客队。帧区间均为左闭右开 `[start_frame, end_frame)`。

### 检测与跟踪参数

演示配置现用 `imgsz: 1280`、`confidence: 0.1`、`nms_iou: 0.5`。检测下限和轨迹激活阈值分开：`0.1 < 分数 < 0.25` 的框参与已有轨迹的第二次关联，新建轨迹仍需达到 `0.35`，避免低分噪声直接创建编号。NMS 在跟踪前抑制重复人物框。

`lost_track_buffer_s: 3.0` 按真实采样帧率换算丢失缓冲；`track_activation_threshold: 0.25` 与 `minimum_matching_threshold: 0.8` 可单独调整。缓冲保留关联状态，漏检帧仍没有真实观测，不能用来补算路程。修改参数后需要重新执行 `track`，新旧运行的编号不能直接互用。

### CLI 报告与历史

```powershell
# 需已确认标定并至少绑定一名球员
& .\.venv\Scripts\python.exe -m footballer_analyzer report --run $runDir

# 显式指定每段素材要采用的报告版本
& .\.venv\Scripts\python.exe -m footballer_analyzer history --player player_001 --reports outputs/run_.../reports/report_...
```

`history --reports` 可接受多个报告目录，同一片段只选一个版本。重叠片段会被拒绝；同场片段偏移未知时只逐片展示。显示的是“已分析片段累计”，不外推整场跑动。

## 产物与结构

```text
configs/                         演示配置、标定点、验证过的依赖版本约束
src/footballer_analyzer/          CLI、检测跟踪、标定、身份、统计、报告、本地服务
src/footballer_analyzer/web/      原生 HTML/CSS/JavaScript 前端，无额外构建步骤
tests/                           时间、分段、身份、重算与 HTTP 边界测试
outputs/run_*/
  manifest.json                  源文件摘要、帧率、配置、依赖、状态与耗时
  tracks.parquet / tracks.mp4     原始观测与轨迹预览
  catalog.json / crops/           轨迹目录与缩略图
  track_catalog.html              可直接打开的缩略图索引
  calibration.yaml               标定、适用范围与确认状态
  bindings.yaml                  本次运行的身份和排除区间
  reports/report_*/              独立版本的统计、图表、视频、HTML 与输入快照
data/players.yaml                跨场球员档案（首次从前端创建）
```

## 验证与当前限制

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check src/footballer_analyzer/web/app.js
```

本轮 **11 项测试通过**。真实 5 秒素材：1080p25、125 帧、1614 条观测、35 条轨迹，跟踪阶段约 10.7 秒（含预采样与产物输出，不含 Python 启动导入）。详细证据见 [08 实现与验收记录](docs/08-实现与验收记录.md)。

当前使用 CPU 检测；调研中的 DirectML 尚未接入本原型。短片可用于检查链路，未证明长片吞吐、跟踪准确率或 ±10% 距离误差。现有模型仍有漏检、断轨和换 ID；报告仅统计可靠区域内的连续有效观测，缺口不插值。首版仅支持恒定帧率、单个固定机位窗口，不包含球、事件、自动 ReID、多摄融合或实时服务。

## 文档与调研工具

- [文档索引](docs/README.md)：需求、选型、调研与专题报告。
- [07 架构设计与最小闭环](docs/07-架构设计与最小闭环.md)：两阶段流水线、数据约束、指标口径与验收标准。
- [08 实现与验收记录](docs/08-实现与验收记录.md)：已实现能力、复现命令、实测与未通过项。
- `scripts/setup_bench_env.ps1`、`bench_cpu.py`、`bench_pipeline.py`：原调研基准工具，不替代上面的应用安装步骤。
- `scripts/inspect_clip.py`：素材体检；`scripts/clone_refs.ps1`：本地获取参考源码。

## Python 代码说明约定

- 遵循 PEP 257，使用中文三双引号 docstring：文件开头说明模块职责，类和函数定义后说明用途。
- 简单函数用一句话；复杂函数补充输入输出、单位、区间、状态或主要异常，避免逐行复述实现。
- 帧号明确来自源视频，区间写为 `[start_frame, end_frame)`；说明位置用像素还是米、时间用秒还是帧。
- 修改算法时同步更新对应说明；文档整理可通过去除 docstring 后比较 AST，检查执行逻辑是否改变。

## 第三方源码与素材

`third_party/`、`data/`、`models/`、`outputs/`、虚拟环境及第三方版权截图均已被 `.gitignore` 排除。仓库不包含视频或模型权重，请使用有权使用的本地素材。第三方许可见 [01 仓库清单与许可](docs/01-参考仓库清单与许可.md)；当前检测依赖 Ultralytics（AGPL-3.0），沿用调研阶段的许可约束。
