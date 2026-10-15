# ============================================================
# 浅克隆「基于摄像头的足球视频分析」参考仓库到 third_party/
#   用法:  pwsh -File scripts/clone_refs.ps1
#   特性:  已存在的目录跳过；单个仓库失败不影响其他；输出汇总表
#   注意:  third_party/ 已在 .gitignore 中，不会被提交
# ============================================================
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'third_party'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$repos = @(
    # ---- 端到端 / 教学级流水线 ----
    @{ n = 'football_analysis';        u = 'https://github.com/abdullahtarek/football_analysis' },   # YOLOv8+ByteTrack+KMeans+光流
    @{ n = 'roboflow-sports';          u = 'https://github.com/roboflow/sports' },                    # pip `sports`：检测/跟踪/球队分类/球场关键点
    @{ n = 'Football-Vision';          u = 'https://github.com/Smasko7/Football-Vision' },            # 微型端到端示例
    @{ n = 'manasvii-Football-Analysis'; u = 'https://github.com/manasvii20/Football-Analysis' },
    @{ n = 'songminkyu-football-analysis'; u = 'https://github.com/songminkyu/football-analysis' },
    @{ n = 'Football-Game-Possession-Tracking'; u = 'https://github.com/Yasouimo/Football-Game-Possession-Traking' },

    # ---- SoccerNet 官方挑战基线（学术 SOTA 方向）----
    @{ n = 'sn-gamestate';             u = 'https://github.com/SoccerNet/sn-gamestate' },             # Game State Reconstruction：跟踪+身份+小地图
    @{ n = 'sn-trackeval';             u = 'https://github.com/SoccerNet/sn-trackeval' },             # 体育跟踪评测指标 (HOTA/DetA/AssA/GS-HOTA)
    @{ n = 'sn-jersey';                u = 'https://github.com/SoccerNet/sn-jersey' },                # 注意：仓库内只有 README（挑战说明），无代码

    # ---- 相机标定 / 单应性（把画面映射到球场坐标）----
    @{ n = 'tvcalib';                  u = 'https://github.com/MM4SPA/tvcalib' },                     # 转播画面自标定
    @{ n = 'PnLCalib';                 u = 'https://github.com/mguti97/PnLCalib' },                   # 点线自标定 (NeurIPS'23)
    @{ n = 'No-Bells-Just-Whistles';   u = 'https://github.com/mguti97/No-Bells-Just-Whistles' },      # 自监督标定+跟踪+事件

    # ---- 跟踪 / ReID 工具箱 ----
    @{ n = 'boxmot';                   u = 'https://github.com/mikel-brostrom/boxmot' },              # 一站式多目标跟踪+ReID
    @{ n = 'deep-person-reid';         u = 'https://github.com/KaiyangZhou/deep-person-reid' },       # torchreid / OSNet
    @{ n = 'SportsMOT';                u = 'https://github.com/MCG-NJU/SportsMOT' },                  # 体育多目标跟踪数据集+基线

    # ---- 数据/可穿戴融合 ----
    @{ n = 'SoccerTrack';              u = 'https://github.com/AtomScott/SoccerTrack' },              # 视频+GNSS 同步数据集

    # ---- 第二批：补充调研发现的可用工程 ----
    @{ n = 'tracklab';                 u = 'https://github.com/TrackingLaboratory/tracklab' },        # sn-gamestate 依赖的跟踪框架
    @{ n = 'tryolabs-soccer-video-analytics'; u = 'https://github.com/tryolabs/soccer-video-analytics' },
    @{ n = 'mradovic38-football-analysis'; u = 'https://github.com/mradovic38/football-analysis' },   # 世界坐标+Voronoi+控球+CSV
    @{ n = 'Football-Analytics-with-DL-and-CV'; u = 'https://github.com/Hmzbo/Football-Analytics-with-Deep-Learning-and-Computer-Vision' },
    @{ n = 'sportsfield_release';      u = 'https://github.com/vcg-uvic/sportsfield_release' },       # 真正的球场配准仓库
    @{ n = 'sn-teamspotting';          u = 'https://github.com/SoccerNet/sn-teamspotting' },
    @{ n = 'Real-Time-Football-Detection'; u = 'https://github.com/rustyneuron01/Real-Time-Football-Detection' }
)

$rows = @()
foreach ($r in $repos) {
    $target = Join-Path $dest $r.n
    $start = Get-Date
    if (Test-Path (Join-Path $target '.git')) {
        Write-Output "[SKIP] $($r.n) 已存在"
        $rows += [pscustomobject]@{ repo = $r.n; status = 'skip'; mb = ''; sec = '' }
        continue
    }
    Write-Output "[CLONE] $($r.n) <- $($r.u)"
    $out = git clone --depth 1 --quiet $r.u $target 2>&1
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        $mb = [math]::Round(((Get-ChildItem -Recurse -Force -File $target -ErrorAction SilentlyContinue |
              Measure-Object -Property Length -Sum).Sum / 1MB), 1)
        Write-Output "  [OK] $($r.n)  $mb MB"
        $rows += [pscustomobject]@{ repo = $r.n; status = 'ok'; mb = $mb; sec = [math]::Round(((Get-Date) - $start).TotalSeconds, 1) }
    } else {
        Write-Output "  [FAIL] $($r.n) rc=$code : $out"
        $rows += [pscustomobject]@{ repo = $r.n; status = 'fail'; mb = ''; sec = [math]::Round(((Get-Date) - $start).TotalSeconds, 1) }
    }
}

Write-Output "`n===== 汇总 ====="
$rows | Format-Table -AutoSize
Write-Output "OK: $(($rows | Where-Object status -eq 'ok').Count)  SKIP: $(($rows | Where-Object status -eq 'skip').Count)  FAIL: $(($rows | Where-Object status -eq 'fail').Count)"
