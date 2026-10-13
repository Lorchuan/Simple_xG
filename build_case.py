"""Build the illustrated case, standalone demo and executable notebook."""
from pathlib import Path
import argparse
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Arc
import joblib
import nbformat
from nbclient import NotebookClient
import markdown
from importlib.metadata import version
from main import ROOT, geometry

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
ASSETS=ROOT/'assets'
ASSETS.mkdir(exist_ok=True)

def pitch(ax):
    ax.set_facecolor('#f0f6f0')
    for patch in [Rectangle((60,0),60,80,fill=False),Rectangle((102,18),18,44,fill=False),
                  Rectangle((114,30),6,20,fill=False),Arc((108,40),20,20,theta1=127,theta2=233)]:
        patch.set_edgecolor('#799887');ax.add_patch(patch)
    ax.plot([120,120],[36,44],lw=4,color='#283d35')
    ax.scatter([108],[40],s=10,color='#799887')
    ax.set(xlim=(60,122),ylim=(0,80),aspect='equal',xlabel='纵向坐标 x',ylabel='横向坐标 y')

def figures(summary,test,model):
    from PIL import Image
    logo=Image.open(ASSETS/'statsbomb_logo.png')
    logo.thumbnail((300,100))
    logo.save(ASSETS/'statsbomb_credit.png')
    fig,ax=plt.subplots(figsize=(6.2,5));pitch(ax)
    ax.scatter([104],[52],c='#c03e47',s=65,zorder=5)
    ax.plot([104,120],[52,40],color='#c03e47',lw=2,label='到球门中心的距离')
    ax.plot([104,120],[52,36],color='#317f9f');ax.plot([104,120],[52,44],color='#317f9f',label='两门柱形成的射门角')
    ax.annotate('射门位置 (104,52)',(104,52),xytext=(77,63),arrowprops={'arrowstyle':'->'})
    ax.legend(loc='lower left',fontsize=9);ax.set_title('图 1  射门位置、距离与角度');fig.tight_layout();fig.savefig(ASSETS/'01_geometry.png');plt.close(fig)
    xx,yy=np.meshgrid(np.linspace(65,119,130),np.linspace(1,79,150))
    d,a=geometry(xx.ravel(),yy.ravel())
    samples=pd.DataFrame({'distance':d,'angle':a,'body_part':'Right Foot','shot_type':'Open Play','under_pressure':0,'first_time':0})
    zz=model.predict_proba(samples)[:,1].reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(6.2,5));pitch(ax)
    im=ax.pcolormesh(xx,yy,zz,cmap='YlGnBu',vmin=0,vmax=.65,shading='auto',alpha=.85)
    fig.colorbar(im,ax=ax,label='预测 xG');ax.set_title('图 2  已训练模型的概率曲面');fig.tight_layout();fig.savefig(ASSETS/'02_probability.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,label,color in [('context','原始模型','#317f9f'),('calibrated','验证集校准后','#c03e47')]:
        r=pd.read_csv(ROOT/f'results/calibration_{name}.csv')
        axes[0].plot(r.mean_xg,r.goal_rate,'o-',label=label,color=color)
        if name=='context': axes[0].errorbar(r.mean_xg,r.goal_rate,yerr=np.array([r.goal_rate-r.wilson_low,r.wilson_high-r.goal_rate]),fmt='none',ecolor=color,capsize=3,alpha=.6)
    axes[0].plot([0,.55],[0,.55],'--',color='#777');axes[0].set(xlabel='分组平均 xG',ylabel='分组实际进球率',title='图 3a  测试集校准图');axes[0].legend(fontsize=9)
    from sklearn.metrics import roc_curve
    fpr,tpr,_=roc_curve(test.goal,test.xg)
    axes[1].plot(fpr,tpr,color='#317f9f',label=f"AUC = {summary['metrics']['context']['auc']:.3f}")
    axes[1].plot([0,1],[0,1],'--',color='#777');axes[1].set(xlabel='假阳性率',ylabel='真阳性率',title='图 3b  测试集 ROC');axes[1].legend()
    fig.tight_layout();fig.savefig(ASSETS/'03_validation.png');plt.close(fig)
    mid=test.groupby('match_id').size().idxmax();match=test[test.match_id==mid]
    fig,ax=plt.subplots(figsize=(6.5,5));pitch(ax)
    for goal,marker,label,col in [(0,'o','未进球','#317f9f'),(1,'*','进球','#c03e47')]:
        r=match[match.goal==goal];ax.scatter(r.x,r.y,s=30+500*r.xg,marker=marker,c=col,alpha=.8,label=label,edgecolors='white')
    ax.legend(loc='lower left');ax.set_title(f"图 4  {match.iloc[0]['home']} vs {match.iloc[0]['away']}\n{match.iloc[0]['date']} · 非点球 xG {match.xg.sum():.2f} / 进球 {match.goal.sum()}",fontsize=10)
    fig.tight_layout();fig.savefig(ASSETS/'04_match.png');plt.close(fig)
    return mid

def document(s,mid):
    m=s['metrics'];ci=s['match_bootstrap_95ci'];pen=s['penalties']
    splits='\n'.join(f"| {label} | {v['first_date']} 至 {v['last_date']} | {v['matches']} | {v['nonpenalty_shots']} | {v['nonpenalty_goals']} |" for key,label in [('train','训练集'),('validation','验证集'),('test','测试集')] for v in [s['split_summary'][key]])
    scores='\n'.join(f"| {label} | {m[key]['log_loss']:.4f} | {m[key]['brier']:.4f} | {m[key]['auc']:.4f} | {m[key]['mean_xg']:.3%} |" for key,label in [('constant','固定进球率基线'),('geometry','距离 + 角度'),('context','距离 + 角度 + 射门条件（主模型）'),('context_calibrated','主模型 + Platt 校准')])
    text=rf'''# 一次射门有多大概率进球？——足球 xG 模型的建立与验证

## 关键信息

| 项目 | 内容 |
| --- | --- |
| 涉及的概率论知识点 | 条件概率、伯努利分布、条件期望、期望的线性性、最大似然估计；拓展：贝叶斯估计与概率校准 |
| 实际应用场景 | 用射门发生时的位置和条件估计进球概率，评价机会质量；结合公开比赛数据训练并验证足球 xG 模型 |
| 延伸的专业课程知识 | 逻辑回归、特征编码、正则化、训练/验证/测试划分、概率预测评价、数值优化 |

完成日期：2026 年 10 月 13 日。适用对象：已学习条件概率、随机变量及数学期望的同学。建议安排：原理讲解 25 分钟，训练与验证 25 分钟，交互实验与讨论 10 分钟。

## 案例说明

足球比赛中，射门次数相同的两支球队，获得的机会未必同样好。xG（expected goals，预期进球）给每次射门分配一个进球概率：例如 xG 为 0.20，表示模型估计这种射门条件下的进球概率为 20%。

本案例用 StatsBomb Open Data 中 **2015/16 赛季英超**的实际射门数据，自行训练一个可解释的逻辑回归模型。目标不是复刻商业 xG，而是完成“概率定义—数据建模—参数估计—独立验证”的过程。

## Problem Setup：把射门写成概率问题

### 1. xG 是什么条件下的概率？

令随机变量 $Y$ 表示一次射门的结果，进球时 $Y=1$，否则 $Y=0$。令 $X$ 表示射门条件，包括距离、角度、射门部位、射门类型、是否受压和是否直接射门。定义

$$
p(x)=P(Y=1\mid X=x),\qquad Y\mid X=x\sim\operatorname{{Bernoulli}}(p(x)).
$$

因此 $E[Y\mid X=x]=p(x)$，$\operatorname{{Var}}(Y\mid X=x)=p(x)(1-p(x))$。一次射门的 xG 就是条件进球概率，也等于条件期望。

模型输出的是对 $p(x)$ 的估计 $\hat p(x)$。它依赖所选特征与训练样本，不能理解为某个球员这次射门的“真实概率”。射门之后的轨迹、门将扑救结果和球的终点，不属于本案例的条件信息。

对于连续的射门位置，$P(X=x)$ 通常为零。这里的条件概率通过条件分布定义；实践中用相似射门的信息和模型假设进行估计，不能直接套用 $P(A\cap B)/P(B)$ 对精确坐标作频数计算。

### 2. 为什么把距离和角度作为特征？

StatsBomb 使用 $120\times80$ 的场地坐标，进攻球门中心为 $(120,40)$，两门柱为 $(120,36)$、$(120,44)$。这些是标准化坐标单位，**不是直接以米计量**。

设射门位置为 $s=(x,y)$，门柱为 $g_1,g_2$，则

$$
d=\sqrt{{(120-x)^2+(40-y)^2}},\qquad
\alpha=\operatorname{{atan2}}\left(|(g_1-s)\times(g_2-s)|,(g_1-s)\cdot(g_2-s)\right).
$$

二维叉积取其标量绝对值，角度 $\alpha$ 以弧度表示。距离相近的射门也可能具有不同角度，所以仅用距离会丢失位置的信息。

![射门几何特征](assets/01_geometry.png)

## Methodology：从条件概率到训练模型

### 为什么不把射门后的质量作为条件？

需要区分“机会质量”与“执行质量”。距离、角度、防守压力等描述射门发生时的机会，本来就是 xG 的条件。球踢出后实际的球速、落点和是否射正描述执行结果；如果加入这些信息，估计的是射门后的进球概率，通常称为 post-shot xG，而不是本案例的射门前 xG。

令 $X$ 为射门前条件，$Q$ 为射门后的执行信息，两种模型分别估计 $P(Y=1\mid X)$ 和 $P(Y=1\mid X,Q)$。它们由全概率公式联系：

$$
P(Y=1\mid X)=E\left[P(Y=1\mid X,Q)\mid X\right].
$$

射门前模型将尚未发生的执行结果平均掉，适合评价获得的机会；射门后模型还评价实际踢得怎样。同一机会的一次射门偏出、另一次射向死角，射门前 xG 可以相同，射门后估计则不同。后者并非不能建模，但射门前尚不可用的特征会造成这个预测任务的时间泄漏；若直接把“是否进球”或据结果打出的质量评分作为输入，还会发生标签泄漏。

### 1. 用逻辑回归表示条件概率

把连续特征标准化，把射门部位与类型编码为独热向量，得到特征向量 $\phi(x)$。逻辑回归设定

$$
\hat p_\theta(x)=\sigma(\theta_0+\theta^\mathsf{{T}}\phi(x)),\qquad
\sigma(z)=\frac{{1}}{{1+e^{{-z}}}}.
$$

等价地，$\log\frac{{\hat p}}{{1-\hat p}}=\theta_0+\theta^\mathsf{{T}}\phi(x)$，即对数胜算随特征线性变化。Sigmoid 将实数映射到 $(0,1)$；它成为进球概率模型，还需要伯努利标签、相应的训练目标和数据验证。仅将数值限制在 $[0,1]$ 内不足以证明它具有正确的概率含义。

### 2. 为什么最小化对数损失？

对射门样本 $(x_i,y_i)$，伯努利模型给出的似然为

$$
L(\theta)=\prod_{{i=1}}^n \hat p_i^{{y_i}}(1-\hat p_i)^{{1-y_i}},\qquad
\hat p_i=\hat p_\theta(x_i).
$$

取负对数并除以样本数，得到二分类交叉熵，也称对数损失：

$$
\ell(\theta)=-\frac1n\sum_{{i=1}}^n\left[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\right].
$$

最小化它对应最大似然估计。训练还加入 $L_2$ 正则化，限制参数过大；代码用验证集选择参数 $C$，$C$ 越小，正则化越强。数值求解交给 scikit-learn 的优化器。

上式采用样本条件独立的工作假设。实际同场比赛的射门可能相关，因此评估置信区间时按整场比赛重抽样，而不是把每次射门都当作独立观测。

### 3. 数据处理与验证规则

原始数据共 {s['total_valid_shots']:,} 次有效射门，其中非点球 {s['nonpenalty_shots']:,} 次，点球 {s['penalty_shots']} 次。主模型只训练非点球；排除无效坐标和点球大战事件。进球标签来自 `shot.outcome`，只用于构建 $Y$，不进入特征。

| 特征 | 处理方式 |
| --- | --- |
| 射门距离、射门角度 | 按训练集均值和标准差标准化 |
| 射门部位、射门类型 | 独热编码；测试中新类别采用全零编码 |
| 受压、直接射门 | 0/1 指示变量；按数据规范，未标注视为否 |

不使用官方 `statsbomb_xg`，不使用射门终点或结果类字段，也不对少数进球样本过采样。否则可能发生答案泄漏，或改变原始进球比例，影响概率估计。

按比赛日期的先后，约以 60%/20%/20% 的**比赛日**划分数据，同一天和同一场比赛不会跨集合。这不是按射门数量精确划分。

| 集合 | 日期范围 | 比赛数 | 非点球射门数 | 非点球进球数 |
| --- | --- | ---: | ---: | ---: |
{splits}

训练集拟合标准化、编码器和模型；验证集在 $C\in\{{0.01,0.1,1,10\}}$ 中选择对数损失最低者；测试集只用于最终评价。另在验证集上拟合 Platt 校准器，作为预先设置的对照，不依据测试结果重新选模型。它将原始对数胜算 $z$ 再映射为 $\sigma(az+b)$，只调整概率尺度，不重新拟合射门特征。

### 4. 怎样验证预测的概率？

**对数损失与 Brier 分数越小越好。** Brier 分数是 $\frac1n\sum_i(\hat p_i-y_i)^2$。若条件进球概率为 $p(x)$，预测值为 $q$，则

$$
E[(q-Y)^2\mid X=x]=(q-p(x))^2+p(x)(1-p(x)).
$$

右边第二项与 $q$ 无关，因此期望 Brier 损失在 $q=p(x)$ 时最小。这解释了为什么概率预测也可以用平方误差评价。

**AUC 衡量排序，校准图检查概率大小。** AUC 可理解为随机抽取一例进球与一例未进球，模型给进球例更高分的概率（同分计半）。它不能单独证明预测的 20% 对应约 20% 的实际进球率。校准图将测试射门按预测概率分为 8 个等样本量组，比较每组平均 xG 与实际进球率。

## Result：实际训练与验证结果

### 1. 模型比较

以下结果由随附脚本实际训练并生成；固定进球率基线使用训练集非点球进球率 {s['train_goal_rate']:.2%}。主模型选择的 $C={s['selected']['context']['C']:g}$。

| 模型 | 对数损失 ↓ | Brier ↓ | AUC ↑ | 平均 xG |
| --- | ---: | ---: | ---: | ---: |
{scores}

测试集实际非点球进球率为 {m['context']['goal_rate']:.2%}。主模型的 AUC 为 {m['context']['auc']:.3f}；按比赛进行 1,000 次 Bootstrap 后，95% 百分位区间为 [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]。相对常数基线的 Brier 改善为 {m['constant']['brier']-m['context']['brier']:.4f}，区间为 [{ci['brier_gain_vs_constant'][0]:.4f}, {ci['brier_gain_vs_constant'][1]:.4f}]。

如果把所有射门都判为“不进球”，准确率仍有 {1-m['context']['goal_rate']:.2%}，但无法区分机会质量。主模型平均 xG 为 {m['context']['mean_xg']:.2%}，低于测试集实际进球率；总预测为 {m['context']['sum_xg']:.1f}，实际进球 {m['context']['goals']}。这提示应检查概率尺度，而不是只看排序指标。上述 Bootstrap 区间反映固定模型在本测试比赛上的评估波动，不包含重新采样训练数据带来的参数不确定性。

![概率曲面](assets/02_probability.png)

图 2 固定为右脚、运动战、不受压、非直接射门，只改变位置。它表示模型的估计，不是所有位置的真实进球概率；样本稀少的位置可能有较大误差。

![校准与排序验证](assets/03_validation.png)

图 3a 的误差条为分组进球率的 Wilson 95% 区间，是二项近似，未处理组内的比赛相关性。它用于辅助观察，不等同于整体模型的置信区间。校准后的模型可能在新数据上变好，也可能变差，应同时报告原始模型与校准模型的结果。

本次校准后的对数损失和 Brier 分数均略高于原始模型，AUC 不变。校准映射的斜率为正，因此保持排序；验证集上的概率调整未改善后续测试集，可能与有限样本和时间变化有关，不能仅据此确定原因。

### 2. 怎样解释一场比赛的 xG？

给定已发生的 $m$ 次射门及其特征，令 $G=\sum_jY_j$。由期望的线性性，

$$
E[G\mid X_1=x_1,\ldots,X_m=x_m]
=\sum_{{j=1}}^m P(Y_j=1\mid X_1=x_1,\ldots,X_m=x_m)
\approx\sum_{{j=1}}^m\hat p(x_j).
$$

线性性本身不要求各次射门独立；最后一步还假设其他射门的信息对本次条件进球概率的影响可忽略。这是对已发生射门的事后机会评价，不是赛前的预期进球预测。

![测试比赛射门分布](assets/04_match.png)

图 4 选取测试集中非点球射门次数最多的比赛，圆点大小对应 xG，星号表示进球。实际进球数与 xG 总和可以不同；单场偏差不能直接判断模型是否有效。

若额外假设射门结果条件独立，可进一步得到 $\operatorname{{Var}}(G\mid X)\approx\sum_j\hat p_j(1-\hat p_j)$ 及“至少一球”的概率 $1-\prod_j(1-\hat p_j)$。足球中的补射、比分变化等会影响这个假设，不能无条件使用。

### 3. 点球与贝叶斯估计（拓展）

点球位置几乎固定，单独建模更清楚。训练集有 {pen['train_n']} 次点球，{pen['train_goals']} 次进球。取 $p_{{\mathrm{{pen}}}}\sim\operatorname{{Beta}}(1,1)$，观察 $n$ 次中 $k$ 次进球后，后验为 $\operatorname{{Beta}}(k+1,n-k+1)$，后验预测进球概率为

$$
E[p_{{\mathrm{{pen}}}}\mid D]=\frac{{k+1}}{{n+2}}={pen['beta_1_1_posterior_mean']:.3f}.
$$

测试集中有 {pen['test_n']} 次点球，{pen['test_goals']} 次进球。样本量较小，不能据此精确区分球员能力；交互 demo 的点球模式采用上述固定估计。

### 4. 亲手完成的实验

打开 `01_xG模型训练与验证.ipynb`，逐格执行数据检查、特征构造、模型拟合与测试评价。Notebook 随附已经执行的输出，并展示真实测试射门的预测值。

打开 `交互demo.html`，改变射门位置、部位和场景，观察同一位置下条件概率如何变化。它使用本案例训练出的参数，不调用官方 xG 或在线接口。再选择测试集中的真实射门，将预测概率与其结果一起查看；一次成功或失败不构成对概率值的检验。

课堂讨论：为什么全猜“不进球”也有高准确率？两个模型 AUC 相同，概率是否一定同样可信？一场比赛 xG 为 2.0，为什么不表示一定进两球？

### 5. 适用范围与局限

本模型没有使用门将位置、防守球员分布、传球过程或球员能力，逻辑回归也限制了特征的非线性作用。数据仅来自一个联赛的一个赛季，时间划分验证仍不能证明它能迁移到其他联赛。本案例不声称达到 StatsBomb 商业模型的效果。

## 代码链接与复现

- [已执行的训练 Notebook](01_xG模型训练与验证.ipynb)
- [数据下载、训练和验证脚本](main.py)
- [交互演示](交互demo.html)
- [完整实验记录](results/summary.json) · [测试预测](results/test_predictions.csv)
- [数据来源与校验记录](data/source_manifest.json) · [运行说明](README.md)

```powershell
python -m pip install -r requirements.txt
python main.py train
```

本机目录内已保留提取后的射门 CSV，上述命令可离线训练。首次在其他机器使用时，运行 `python main.py all` 从固定版本的官方公开仓库下载比赛事件并提取数据。为遵守数据许可，交付 ZIP 不再分发原始事件、射门 CSV 和逐条测试数据。随机种子为 {s['seed']}，实际环境版本见实验记录。所有正文公式保留为可编辑 LaTeX；飞书导入后若公式未自动识别，可粘贴到原生公式块，勿转为图片。

## 参考资料

1. [StatsBomb Open Data（Hudl 官方仓库）](https://github.com/hudl/open-data)：比赛事件数据、许可与字段规范。
2. [Hudl / StatsBomb：Expected Goals Explained](https://www.hudl.com/blog/expected-goals-xg-explained)：xG 的定义与常见影响因素。
3. [scikit-learn：LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)：模型与正则化参数。
4. [scikit-learn：Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)：校准曲线与概率校准。
5. [StatsBomb：Upgrading Expected Goals](https://statsbomb.com/articles/soccer/upgrading-expected-goals/)：机会质量与包含落点、球速的执行质量模型。

数据来源：StatsBomb Open Data。使用和再发布须遵守附带的 `data/LICENSE.pdf`；对外发布需按官方说明注明 StatsBomb 并使用其标识。本案例借助 AI 辅助编写讲解与代码；指标和图表来自实际训练、测试与程序检查，未填造实验结果。

![StatsBomb 数据来源标识](assets/statsbomb_credit.png)
'''
    (ROOT/'案例文档.md').write_text(text,encoding='utf-8')
    # Protect TeX from Markdown backslash escaping; MathJax renders editable text.
    import re,html
    formulas=[]
    def keep(match):
        formulas.append(match.group(0));return f'FORMULATOKEN{len(formulas)-1}END'
    protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',keep,text)
    body=markdown.markdown(protected,extensions=['tables','fenced_code'])
    for i,f in enumerate(formulas):body=body.replace(f'FORMULATOKEN{i}END',html.escape(f))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>足球 xG 概率论案例</title><style>body{max-width:920px;margin:40px auto;padding:0 24px;font:16px/1.85 "Microsoft YaHei",sans-serif;color:#202b2c}h1{font-size:28px}h2{font-size:23px;border-bottom:1px solid #ddd;padding-top:20px}h3{font-size:19px}img{max-width:100%;display:block;margin:20px auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#edf4f2}pre{background:#f3f4f4;padding:16px;overflow:auto}a{color:#136e7b}.MathJax{max-width:100%;overflow-x:auto}@media print{body{margin:0}img{max-height:350px}h2,h3{break-after:avoid}table,img{break-inside:avoid}}</style><script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script><body>'''+body+'</body></html>'
    (ROOT/'案例文档.html').write_text(page,encoding='utf-8')

def demo(test,publish=False):
    params=json.loads((ROOT/'models/browser_model.json').read_text(encoding='utf-8'))
    examples=[] if publish else test.sample(40,random_state=42)[['x','y','body_part','shot_type','under_pressure','first_time','goal','xg','home','away','minute']].to_dict('records')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>xG 射门实验</title><style>*{box-sizing:border-box}body{margin:0;background:#f7f9f8;color:#253730;font:15px "Microsoft YaHei",sans-serif}main{max-width:1000px;margin:auto;padding:24px}header{border-bottom:1px solid #cbd9d1;margin-bottom:24px}h1{font-size:26px}.layout{display:grid;grid-template-columns:1fr 290px;gap:24px}canvas{width:100%;aspect-ratio:62/80;display:block;background:#e6efe9;touch-action:none}label{display:block;margin:15px 0 5px}select,input[type=number]{width:100%;padding:9px;border:1px solid #a9beb0;border-radius:4px;background:white}input[type=checkbox]{margin-right:8px}output{display:block;font-size:44px;color:#176c55;margin:18px 0}button{padding:10px 16px;background:#176c55;color:white;border:0;border-radius:4px;cursor:pointer}small{display:block;color:#5a6b63;line-height:1.7}.coords{display:grid;grid-template-columns:1fr 1fr;gap:10px}#example{font-size:14px;line-height:1.7;margin-top:14px}footer{margin-top:24px;border-top:1px solid #cbd9d1;padding-top:12px;font-size:12px}@media(max-width:650px){.layout{grid-template-columns:1fr}main{padding:16px}h1{font-size:23px}canvas{max-height:460px}}</style><main><header><h1>足球 xG · 射门实验</h1><p>2015/16 英超 · 实际训练的逻辑回归模型</p></header><div class="layout"><canvas id="pitch" width="620" height="800" aria-label="射门位置" tabindex="0"></canvas><section><label>预测进球概率</label><output id="prob"></output><div class="coords"><div><label for="x">x 坐标</label><input id="x" type="number" min="60" max="119.9" step="0.1" value="108"></div><div><label for="y">y 坐标</label><input id="y" type="number" min="0" max="80" step="0.1" value="40"></div></div><label for="body">射门部位</label><select id="body"></select><label for="type">射门类型</label><select id="type"></select><label><input id="pressure" type="checkbox">受压</label><label><input id="first" type="checkbox">直接射门（first time）</label><small id="geometry"></small><label for="examples">测试集真实射门</label><select id="examples"><option value="">自定义</option></select><div id="example"></div></section></div><footer>数据：StatsBomb Open Data · 主模型不含点球；点球模式使用训练数据的 Beta(1,1) 后验预测。场地为标准化坐标，模型适用范围与验证结果见案例文档。</footer></main><script>const M=MODELJSON,E=EXAMPLEJSON;const $=id=>document.getElementById(id),c=$('pitch'),ctx=c.getContext('2d');for(const [id,cats] of [['body',M.categories[0]],['type',[...M.categories[1],'Penalty']]])for(const t of cats){const o=new Option(t,t);$(id).add(o)}$('body').value='Right Foot';$('type').value='Open Play';E.forEach((e,i)=>$('examples').add(new Option(`${e.home} vs ${e.away} · ${e.minute}′`,i)));let point={x:108,y:40};
function predict(x,y,b,t,u,f){const dx=120-x,d=Math.hypot(dx,y-40),a=Math.atan2(8*Math.abs(dx),dx*dx+(36-y)*(44-y));let z=M.intercept,k=0;[d,a].forEach((v,j)=>z+=(v-M.numeric_mean[j])/M.numeric_scale[j]*M.coef[k++]);[b,t].forEach((v,j)=>M.categories[j].forEach(cat=>z+=(v===cat?1:0)*M.coef[k++]));z+=u*M.coef[k++]+f*M.coef[k++];return {p:t==='Penalty'?M.penalty_probability:1/(1+Math.exp(-z)),d,a}}
function line(points,color='#6c927b',width=2){ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo((x-60)*10,y*10):ctx.moveTo((x-60)*10,y*10));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(){ctx.clearRect(0,0,620,800);for(let i=0;i<6;i++){ctx.fillStyle=i%2?'#e1ece4':'#e9f1ec';ctx.fillRect(i*100,0,100,800)}line([[60,0],[120,0],[120,80],[60,80],[60,0]]);line([[120,18],[102,18],[102,62],[120,62]]);line([[120,30],[114,30],[114,50],[120,50]]);line([[120,36],[120,44]],'#203d2c',7);line([[point.x,point.y],[120,36]],'#4981a0',2);line([[point.x,point.y],[120,44]],'#4981a0',2);ctx.beginPath();ctx.arc((point.x-60)*10,point.y*10,9,0,Math.PI*2);ctx.fillStyle='#c03e47';ctx.fill()}
function update(){point.x=Math.max(60,Math.min(119.9,Number($('x').value)));point.y=Math.max(0,Math.min(80,Number($('y').value)));const r=predict(point.x,point.y,$('body').value,$('type').value,+$('pressure').checked,+$('first').checked);$('prob').textContent=(100*r.p).toFixed(1)+'%';$('geometry').textContent=`距离 ${r.d.toFixed(1)} 坐标单位 · 射门角 ${(r.a*180/Math.PI).toFixed(1)}°`;$('x').disabled=$('y').disabled=$('body').disabled=$('pressure').disabled=$('first').disabled=$('type').value==='Penalty';if($('type').value==='Penalty'){point={x:108,y:40};$('x').value=108;$('y').value=40;$('geometry').textContent='点球固定位置 · 不区分球员能力'}draw()}
['x','y','body','type','pressure','first'].forEach(id=>$(id).addEventListener('input',()=>{$('examples').value='';$('example').textContent='';update()}));let down=false;c.addEventListener('pointerdown',ev=>{down=true;c.setPointerCapture(ev.pointerId);move(ev)});c.addEventListener('pointermove',ev=>{if(down)move(ev)});c.addEventListener('pointerup',()=>down=false);function move(ev){if($('type').value==='Penalty')return;const b=c.getBoundingClientRect();$('x').value=(60+(ev.clientX-b.left)/b.width*62).toFixed(1);$('y').value=((ev.clientY-b.top)/b.height*80).toFixed(1);$('examples').value='';$('example').textContent='';update()}
$('examples').addEventListener('change',()=>{if($('examples').value==='')return;const e=E[+$('examples').value];$('x').value=e.x;$('y').value=e.y;$('body').value=e.body_part;$('type').value=e.shot_type;$('pressure').checked=!!e.under_pressure;$('first').checked=!!e.first_time;update();$('example').textContent=`真实结果：${e.goal?'进球':'未进球'} · 保存的测试预测 ${(100*e.xg).toFixed(1)}%`});for(const e of E){if(Math.abs(predict(e.x,e.y,e.body_part,e.shot_type,e.under_pressure,e.first_time).p-e.xg)>1e-10)throw Error('模型参数验证失败')}update();</script></html>'''
    page=page.replace('MODELJSON',json.dumps(params,ensure_ascii=False)).replace('EXAMPLEJSON',json.dumps(examples,ensure_ascii=False))
    if publish:
        page=page.replace('<label for="examples">测试集真实射门</label>','<label for="examples" style="display:none">测试集真实射门</label>').replace('<select id="examples">','<select id="examples" style="display:none">')
    (ROOT/'交互demo.html').write_text(page,encoding='utf-8')

def notebook(publish=False):
    nb=nbformat.v4.new_notebook()
    cells=[('m','# 足球 xG：实际训练与验证\n数据：StatsBomb Open Data，2015/16 英超。依次执行以下单元；只从射门发生时的特征估计条件进球概率。'),
    ('c','from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\nfrom main import load_data, make_model, evaluate, fit, ROOT\ndf, split_info = load_data()\nprint(split_info)\ndf.groupby("split").agg(射门数=("goal","size"), 进球数=("goal","sum"), 比赛数=("match_id","nunique"))'),
    ('m','## 1. 条件概率与特征\n$Y\\mid X=x\\sim\\operatorname{Bernoulli}(p(x))$。距离和角度从射门坐标计算；编码与标准化只在训练集拟合。'),
    ('c','df[["x","y","distance","angle","body_part","shot_type","under_pressure","first_time","goal"]].head(8)'),
    ('m','## 2. 亲手拟合一个模型\n主模型排除点球。先拟合 $C=1$ 的逻辑回归，再运行完整的验证集选参、校准与测试流程。'),
    ('c','shots = df[df.shot_type != "Penalty"]\ntrain = shots[shots.split == "train"]\nval = shots[shots.split == "validation"]\nmodel = make_model(C=1, context=True)\nmodel.fit(train, train.goal)\nevaluate(val.goal, model.predict_proba(val)[:,1])'),
    ('c','encoder = model.named_steps["features"]\npd.DataFrame({"特征":encoder.get_feature_names_out(),"系数":model.named_steps["model"].coef_[0]})'),
    ('m','## 3. 完整训练与独立验证\n以下真正重新拟合候选模型，使用验证集选择参数，再生成测试结果和按比赛重抽样的区间。'),
    ('c','fit()\nsummary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))\npd.DataFrame(summary["metrics"]).T'),
    ('c','pred = pd.read_csv(ROOT / "results/test_predictions.csv")\nassert set(train.match_id).isdisjoint(pred.match_id)\nassert train.date.max() < val.date.min() < pred.date.min()\nprint("不同集合无比赛重叠，时间顺序检查通过")\npred[["home","away","minute","distance","angle","goal","xg"]].head(12)'),
    ('m','## 4. 图形验证\nAUC 看排序，校准图检查概率大小。分组进球率有抽样波动，不能要求每个点恰好落在对角线上。'),
    ('c','from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / "assets/03_validation.png")))\nsummary["match_bootstrap_95ci"]'),
    ('m','## 5. xG 总和与真实进球\n期望具有线性性。按射门自身特征估计的概率总和，是比赛机会质量的近似评价；它不是承诺实际进球数。'),
    ('c','pred.groupby(["match_id","home","away"]).agg(射门数=("goal","size"), 实际进球=("goal","sum"), 非点球xG=("xg","sum")).head(12)'),
    ('m','## 6. 点球的贝叶斯估计\n若先验为 Beta(1,1)，后验预测概率为 $(k+1)/(n+2)$。'),
    ('c','pen = df[(df.shot_type == "Penalty") & (df.split == "train")]\nn, k = len(pen), int(pen.goal.sum())\nprint(f"训练点球：{n} 次，进球 {k} 次；后验预测概率 {(k+1)/(n+2):.3f}")'),
    ('m','## 思考\n- 为什么高分类准确率不足以验证 xG？\n- 校准之后测试损失是否一定下降？\n- 若加入射门终点坐标，预测任务的含义会怎样变化？\n\n数据许可见 data/LICENSE.pdf。AI 辅助编写，实验结果来自实际运行。')]
    nb.cells=[nbformat.v4.new_markdown_cell(t) if kind=='m' else nbformat.v4.new_code_cell(t) for kind,t in cells]
    nb.metadata.kernelspec={'display_name':'Python (xG case)','language':'python','name':'xg-case'}
    client=NotebookClient(nb,timeout=600,kernel_name='xg-case',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    if publish:
        # Publish analysis and aggregate metrics without redistributing event rows.
        for index in [3,9]:
            nb.cells[index].outputs=[]
    nbformat.write(nb,ROOT/'01_xG模型训练与验证.ipynb')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true',help='Omit individual shot rows from public artifacts')
    args=parser.parse_args()
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    test=pd.read_csv(ROOT/'results/test_predictions.csv');model=joblib.load(ROOT/'models/xg_model.joblib')['model']
    mid=figures(s,test,model);document(s,mid);demo(test,args.publish);notebook(args.publish)
    packages=['numpy','pandas','scikit-learn','scipy','matplotlib','requests','joblib','nbformat','nbclient','ipykernel','markdown']
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{p}=={version(p)}' for p in packages)+'\n',encoding='utf-8')
    excluded={'shots.csv','shots_features.csv','test_predictions.csv','matches.json','verify_demo.cjs'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'raw_events' not in p.parts and '__pycache__' not in p.parts and 'qa' not in p.parts and p.suffix!='.zip' and p.name not in excluded]
    with zipfile.ZipFile(ROOT/'足球xG案例_完整交付.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    print('Built case, figures, demo, executed notebook and delivery archive.')
