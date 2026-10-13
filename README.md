# 足球 xG 概率论案例

从条件概率、伯努利分布和最大似然估计出发，使用 StatsBomb 2015/16 英超公开数据训练逻辑回归 xG 模型。正文、图表和指标由实际实验生成。

## 文件

- `案例文档.md`：按参考案例结构编写，公式为可编辑 LaTeX。
- `案例文档.html`：图文阅读版，公式显示需要联网加载 MathJax。
- `交互demo.html`：独立运行的实际模型演示，不需要网络或服务器。
- `01_xG模型训练与验证.ipynb`：已执行的训练、验证及课堂实验。
- `main.py`：下载、提取、特征工程、训练、选参、校准、测试及按比赛 Bootstrap。
- `build_case.py`：根据实验结果生成讲义、图表、demo、已执行 Notebook 和交付包。
- `data/shots.csv`、`shots_features.csv`：射门样本及划分；`source_manifest.json`：源版本和每场数据的 SHA256。
- `models`：拟合模型及浏览器参数；`results`：测试预测、指标和校准分组。

## 运行

需要 Python 3.12 或兼容版本。从本仓库首次复现时运行：

```powershell
python -m pip install -r requirements.txt
python main.py all
```

源数据固定到 manifest 内记录的 Git 提交，脚本会在 `data/raw_events` 缓存压缩事件，并生成本地 CSV。此后可用 `python main.py train` 离线重训。为遵守数据许可，本仓库不分发原始事件、射门 CSV、比赛清单和逐条测试数据；复现者从官方公开源自行获取。文档内的分析结果、图表和参数保留。

`requirements-lock.txt` 记录本次实际使用的核心包版本；需要对齐实验环境时用它代替 `requirements.txt` 安装。下载器从同一固定 Git 版本读取数据，并通过 jsDelivr 的 GitHub 镜像应对访问超时，不使用其他第三方整理的数据集。

生成文档并执行 Notebook：

```powershell
python -m ipykernel install --user --name xg-case --display-name "Python (xG case)"
python build_case.py
```

图表生成使用 Windows 微软雅黑字体。交互页直接打开 HTML；可以选点、修改坐标或查看真实测试射门。

公开仓库中的演示省略逐条比赛事件，Notebook 省略逐条射门表格的输出，保留实际训练的指标和图表。自行下载数据后执行 `python build_case.py`，可在本地生成包含真实测试射门的完整演示；对外发布时用 `python build_case.py --publish`。

## 实验约定

主模型仅评估非点球，点球另用 Beta(1,1) 后验预测。按日历日期分割，训练/验证/测试比赛完全分离。类别编码和标准化只拟合训练集，验证集选择正则化参数，测试结果不参与选参。主模型事先指定为含上下文的未校准逻辑回归，校准版本作为对照报告。不使用官方 xG、球的终点或任何射门结果作为输入。

场地是 120×80 的标准化坐标，距离不是米，角度以弧度输入模型。模型不考虑门将位置、防守球员分布或球员能力。

## 飞书导入

导入 `案例文档.md` 时，保留 `assets` 的相对目录结构；若飞书导入器未携带本地图片，请上传对应 PNG 到图号处。公式保留 LaTeX 源码，若未自动渲染，使用飞书原生公式块粘贴，勿截图。交互 HTML 和 Notebook 可作为附件上传。

## 来源与许可

数据：StatsBomb Open Data，官方仓库 https://github.com/hudl/open-data 。许可原文为 `data/LICENSE.pdf`；其中第 1.2.1 条限制向第三方分发数据，第 1.4 条要求公开分析使用 StatsBomb 标识。公开分享案例时使用交付 ZIP，勿将本机的数据缓存和逐条射门 CSV 一并发布。本案例不授予额外的数据权利。

AI 辅助编写讲解和代码，数据指标来自实际运行。
