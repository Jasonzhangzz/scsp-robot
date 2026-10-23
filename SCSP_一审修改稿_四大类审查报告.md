# SCSP 一审修改稿完整审查报告（四大类）

> **对象**：用户上传的 `root.pdf`（20 页修改稿）  
> **审稿材料**：`Review149408(Reviewer3)Of26-0767v1.txt`、`Review151216(Reviewer5)Of26-0767v1.txt`、`AttachmentToReview149406Of26-0767v1.pdf`、`reviewer Editorial Board.docx`。  
> **配套文件**：`SCSP_一审修改稿_高亮批注.pdf`。同一编号（如 `[M03]`）对应 PDF 的高亮批注。  
> **审查性质**：对论文正文、公式、证明、图表和审稿意见进行文本级与数学级审核；**未运行源码或重新计算原始实验日志**，没有拿到独立的原始 arXiv PDF 或正式 Response Letter。不能将“正文已回应”等同于“回复信已逐条填写”，也不能将数据疑点当成确定造假或错误。

## 摘要结论

| 用户要求 | 结论 | 仍需完成的重点 |
|---|---|---|
| 一、是否完整、合理回复全部 Reviewer | **主要意见大多已回应；少数问题没有闭合** | MIQP/CEM/Exchange、Ranking 参数扫描、CPO 终端代价、泛化措辞、复现参数 |
| 二、逻辑自洽、证明有效 | **总体研究框架成立；若干公式与证明尚不严谨** | SCM Eq.(27)/(29)/(31)、Lemma 2、Appendix A、Eq.(34)/(35)/(45)/(46)；Fig.10 与表格数据口径 |
| 三、语言风格是否统一 | **基本统一，不必重写全文** | 技术词义、图内编号、命名、单位及文本-图表一致性；尚未能与单独原版 PDF 做逐句比对 |
| 四、修改是否破坏原始技术逻辑 | **未破坏 CSO→Ranking→CPO 核心路线，但一些修订可能改变实现行为** | 核对 SCM 投影、CEM 求解器、EMA 切换、placing 目标与所有实验代码版本 |

**分类说明**：

- **P0（优先处理）**：直接影响数学定义、关键新增实验证据，或有明显数据错误风险。
- **P1（重要）**：影响 Reviewer 核心问题是否解决、公式与实现统一、技术结论的强度。
- **P2（完善）**：需补充方法、统计口径、术语或排版信息。
- **核实**：现稿引发合理疑问，但必须看代码、原始实验或原始 PDF；不能先下确定性结论。

**PDF 用色**：黄色为 P0，橙色为 P1，浅蓝色为 P2，浅绿色为“核实”；双图外的橙色虚线框只用于定位 Fig.10。点击或点按高亮区域可在支持 PDF 批注的阅读器中查看“问题”和“修改意见”。PDF 原有正文和图表不作任何替换。

---

# 一、修改稿是否回复了所有 Reviewer，且回应是否合理

## 1.1 审稿材料覆盖与总体判断

此处对四份材料逐项核查：Reviewer 3；Review 149406 附件；Reviewer 5；Editorial Board。Editorial Board 的内容与三位 Reviewers 存在交叠，但它强调的要点——Ranking 的反复数学矛盾、MIQP/SNOPT 不符、potential-field 错误、原 Lemma 3/4 的薄弱证明及贡献归因——均独立核对。

**总体判断**：绝大部分审稿意见已被“考虑并修改”，且 Ranking 消融、SCM 经验验证、硬件样本量和可信区间等方面有实质增强；但“有回应”不等于“回应足够”。尤其 Fig.10 对两个参数的独立扫描并未有效提供，MIQP/Exchange 的全文用词尚未闭合。

### Reviewer 3：逐项答复状态

| 审稿意见 | 修改稿响应 | 审查结论 |
|---|---|---|
| ρ/κ/γ 定义自相矛盾 | Eq.(37)–(39) 改写归一化增益、阈值、EMA | **基本解决**；仍需保证 Algorithm 2 与 Eq.(45) 一致 |
| T1–T3 的计数/reset 歧义 | 替换为 EMA | **基本解决**；本身是算法变动，参见第四类 |
| Globally/asymptotically optimal 过强 | 限定为 sampled finite set | **解决** |
| MIQP/QP 与 SNOPT 不匹配 | P8 改述候选枚举 + GPU CEM | **部分解决**；Fig.2、Table IX 仍写 MIQP（R01/R12） |
| Exchange step 未定义 | 采样/筛选伪代码仍称 Exchange | **未解决**（R02/R13） |
| 旋转 100% 与 19/20 冲突 | 现写 119/120（99.2%） | **已回应**；统计需与新硬件试验对应 |
| Long-horizon open-loop 数缺失 | Table X 增加 15/20 vs 20/20 | **解决** |
| Multi-finger 16/20 vs 15/20 不应称显著提升 | 改为较弱的 higher success | **解决** |
| 缺少 confidence intervals | 硬件 Table VII 增加 Wilson CI | **主要解决**；主仿真结果可进一步补充 |
| 单侧 friction constraints | Eq.(31) 改为双侧绝对值限制 | **解决** |
| w_att/w_obs/w_con/w_o/σ/d/solver settings 不足 | P11 增加多个权重及 solver 信息 | **部分解决**；σ/d、CEM 详细设定仍不足（R05） |
| Offline sampling 4.82 s 误导 | Table IX 写 Offline sampling | **解决** |
| 无用 α | P8 仍定义 α>0，但公式未使用 | **未解决**（R04） |
| Q̃ 未定义 | 相关记号基本被修正 | **基本解决** |
| Vf / terminal cost 与公式不符 | Eq.(46) 仍不够明确 | **未完全解决**（M12） |
| 目标函数 free time index | P8/P10 仍混用 i,k,N,T | **部分解决**（M08/M12） |
| Eq.(18) 切平面奇异 | 正文增加备选参考轴 | **基本解决**，算法文本可同步 |
| 姿态成功容差角度 | 给出 14.1° | **文义已回应**；若阈值改变需重新确认（T01） |
| 质量分布不清 | 给出离散值集合 | **部分解决**；各值概率或抽样规则不明确 |

### Review 149406：八项 Major Concerns

| 原始审稿关注 | 修改稿响应 | 审查结论 |
|---|---|---|
| 1. CSO-as-MIQP 数学及 Solver 说明错误 | 使用 candidate enumeration / CEM | **部分解决**（R01/R12/R13） |
| 2. SCM 假设强、缺少高保真误差对比 | Table IV 比较 SCM/ComFree/MuJoCo QP/ContactSDF 与 LCP | **针对性实验已补**，但新数学公式需要再核查（M01–M07） |
| 3. Ranking 内在矛盾 | 重写 ρ/κ + EMA | **主要解决**，伪代码与连续代价待统一（M09/M11） |
| 4. 势场符号与 placing 目标点错误 | Eq.(40) 排斥符号修改，Eq.(44) 使用 p_ref | **核心问题已解决** |
| 5. 各模块贡献无法归因 | 新增 A1–A9 和 Counterfactual | **部分至较充分**；并非每个代价项被独立消融（M15/T02） |
| 6. Baselines 缺失/不公平 | 加入 STOCS-3D、ContactSDF、CMA-ES、MPPI | **有较大改进**，同计算预算与 modified Sampling 仍需说明（R06/R07） |
| 7. Multi-contact/generalization 过度宣称 | Discussion 增加限制 | **部分解决**；P17 仍写 can be easily extended（R09/R10） |
| 8. Hardware 样本量和错误归因不足 | 每条件 30 次、Wilson CI、失败案例 | **主要解决**；请核对 ET 重复与 failure-case 定义（M18/M19） |

其 Technical Corrections 中，双边摩擦约束、四元数误差不再标 rad、最优接触点符号与删除旧 Lemma 4 已有明显修复；关于 smooth、力/冲量量纲、公式和实际 Solver 的一致性，仍建议依本报告第二部分修订。

### Reviewer 5：逐项答复状态

| 原始关注 | 修改稿响应 | 审查结论 |
|---|---|---|
| Ranking 的关键贡献缺少消融 | A1/A2 + Counterfactual | **基本解决** |
| Switch frequency 未统计 | Table VI ~0.2 Hz physical switching | **解决** |
| Ranking 帮助有多大 | normalized advantage | **已提供直接证据**，协议细节可强化（M15） |
| Ranking 参数敏感性 | Fig.10 | **未有效解决：双图横轴和曲线重复（R08）** |
| Vision-only 需承认手动 MuJoCo 校准 | P15/P19 加入 proxy / calibration 与 limitation | **解决主要要求** |
| Introduction 重复、形容词过多 | 重新压缩引言 | **基本解决** |
| Semantic unsuitable regions 未解释 | 补 handle / semantic mask 案例 | **部分解决**：mask 来源及流程未说明（R03） |
| 图例、坐标可读性差 | 图表有所调整 | **基本回应，但应最终检查可读性** |
| Timesteps 的秒数未定义 | P11 明确 h=0.02s | **基本解决** |
| Case Study II 名称与表述 | 改为 Articulated Object Manipulation | **解决** |
| Active CSO 避免 singularity 缺少证据 | P19 仍提 singular configurations | **未充分解决**（R11） |
| CSO 被称为 world model，概念扩张 | 相关措辞基本收紧 | **基本解决**；PointWorld 的未来替换关系需谨慎（L05） |
| Lemma 3 的 LCP perturbation error bound 太宽 | 删除旧 Lemma 3，改为经验接触模型误差对比 | **对原问题的处理方向合理**；不是“新证明都有效”的保证 |
| 和 Jin 的理论创新性界限不清 | 更突出 prior-guided interface / Ranking | **改善明显**，仍可避免过度 SCM 创新主张 |
| 小错：缩写、引用、下标、拼写、采样耗时等 | 大部分修正，Table IX 显示 offline sampling | **大部分解决**；仍需交叉检查所有图号与单位（L01/L04） |

### Editorial Board 的综合要求

| 要求 | 当前判断 |
|---|---|
| Ranking 数学定义错误 | **基本解决**，但离散与连续成本必须一致 |
| MIQP 与实际 Solver 不符 | **部分解决**，图/表未统一 |
| Potential-field 物理方向与 reference cost | **关键错误已修复** |
| 原 Lemma 3/Lemma 4 理论不充分 | **原定理已删除/弱化**；不能因此忽视新增证明错误 |
| 表格中明显数值/成功率矛盾 | **原指定矛盾大体修复**，新版数据还有待核实事项 |
| Ranking 的频率、收益、敏感性 | **前两项有实质新证据，敏感性图需重绘** |
| 强调 Ranking 比简单 surrogate 模型更有贡献 | **改进方向合理** |

## 1.2 仍影响“Reviewer 问题是否真正闭合”的批注项

以下编号对应配套 PDF 中的高亮位置；各项均写明具体修改建议。

### [R01] P5｜P1｜原审稿意见未闭合

**定位**：`Solving MIQP`。

**问题**：Reviewer 3 / Review 149406 / Editorial Board 均质疑 MIQP。P8 已将 CSO 描述为逐候选枚举与 GPU-parallel CEM，但框架图仍写“Solving MIQP”。

**建议**：将 Fig. 2 中算法名称和求解器类型改为与实际代码一致的 finite candidate enumeration + continuous force optimization，并同步核对表 IX。

### [R02] P6｜P1｜原审稿意见未闭合

**定位**：`Sampling and Exchange Method`。

**问题**：Reviewer 3 指出 Exchange step 未定义。Algorithm 1 只列出 ACD、FPS、valid-region selection，没有明确 exchange 操作。

**建议**：若实现无 exchange，请更名为 Sampling and Valid-Region Selection；若有，请写出交换步骤、停止条件和复杂度。

### [R03] P6｜P2｜原审稿意见部分回应

**定位**：`getValidRegion`。

**问题**：Reviewer 5 询问语义禁止区和有效区域怎样确定。当前函数名和例子并未说明掩码由人工标注、语义模型还是规则生成。

**建议**：补充 getValidRegion 的输入、语义 mask 来源、在线/离线属性、是否需要人工标注，以及对比实验中的规则。

### [R04] P8｜P2｜原审稿意见未闭合

**定位**：`α > 0`。

**问题**：Reviewer 3 指出 α 被定义而未使用。这里仍称 α>0 为权重，但 Eq. (32)/(33) 未出现 α。

**建议**：若无实际用途，删除这句 α 定义；如公式漏乘 α，则依据代码补回且更新实验参数表。

### [R05] P11｜P2｜实现参数未完全说明

**定位**：`20 GPU-parallel`。

**问题**：Reviewer 3 要求 solver settings。稿件说明 20 次 GPU-parallel CEM iterations，但未在此说明 population size、采样分布/方差、elite fraction、随机种子等关键配置。

**建议**：补充 CEM 关键超参数和硬件/并行设定；若详见开源代码，可给出准确链接与默认参数表。

### [R06] P11｜P1｜基线公平性仍需论证

**定位**：`We replace its original`。

**问题**：Review 149406 指出 Sampling 原始 contact model 被替换为 f_cf 会产生 modified baseline。当前保留该替换说明，但还需阐明“公平比较”的含义。

**建议**：明确以统一模型比较算法结构，避免声称是完全原版 Sampling；尽可能另列原论文实现或消融，并说明预算。

### [R07] P12｜P2｜计算预算尚未对齐

**定位**：`A7 solves`。

**问题**：Review 149406 要求 MPPI/CEM/CMA-ES 在相同动态模型及 wall-clock budget 下比较。虽然加了 MPPI 类消融，稿中未充分报告匹配预算与优化变量设置。

**建议**：补充优化变量、horizon、rollouts、迭代数、GPU/CPU 与相同时间预算，或明确这只是非严格预算匹配的辅助消融。

### [R08] P14｜P0｜新增回复证据明确错误

**定位**：`Fig. 10. Sensitivity`。

**问题**：Reviewer 5 / Editorial Board 要求 rho_bar 与 beta 的敏感性。Fig.10 左右两图均标“EMA coefficient β”，曲线和样本点也完全相同，未实际展示两个独立参数扫描。

**建议**：核对原始日志；分别重画 rho_bar 扫描和 beta 扫描（注明各自固定参数、试验次数及误差条）；不能只改横轴文字。

### [R09] P17｜P1｜过度宣称仍残留

**定位**：`can be easily extended`。

**问题**：Review 149406 明确要求弱化一般 multi-contact 可扩展性。这里仍称“can be easily extended”，而未证明接触组合数增加时的可扩展复杂度。

**建议**：改为“demonstrate extensions in selected multi-contact case studies”；说明候选组合的增长及实验限制。

### [R10] P18｜P2｜扩展复杂度需交代

**定位**：`top 5 pairs`。

**问题**：多指/双手抓取仅搜索 top-5 抓取区域对及邻近接触，未回答 n_r 个接触从 n_s 候选中组合选取的规模增长问题。

**建议**：说明候选组合的限制、剪枝、运行时间及 n_s/n_r 增长的扩展限制；避免一般多接触规划保证。

### [R11] P19｜P1｜缺少独立证据

**定位**：`approaching singular configurations`。

**问题**：Reviewer 5 特别指出 active CSO 避免 singularities 没有量化依据。新版此处仍将 approaching singular configurations 用作在线 CSO 优势的理由。

**建议**：如果没有关节条件数、奇异失败案例、可达性比较等证据，改为非因果性观察或删除对奇异性的归因。

### [R12] P17｜P1｜全文求解器矛盾

**定位**：`MIQP cost time`。

**问题**：P8 已明确“GPU-parallel CEM”与候选枚举，此表仍列“MIQP cost time”；审稿人正是质疑 MIQP 与实现不符。

**建议**：按真实采样/候选评价/连续优化/通信阶段重新命名计时项目，确保加总口径和 CSO total 一致。

### [R13] P17｜P1｜步骤命名尚未统一

**定位**：`Exchange cost time`。

**问题**：Table IX 仍有“Exchange cost time”，但 Algorithm 1 及正文未定义 exchange 步骤。

**建议**：解释其具体运算（如 KD-tree 最近邻、筛选或排序），并统一 Fig.2/Algorithm 1/计时表；否则删除该命名。

**第一类结论**：不应将这篇修改稿评价为“大多数 Reviewer 意见没回应”。更准确的表述是：**大多数已经针对性修改，少数重要问题尚未完全闭合**。其中“Fig.10 敏感性图重复”和“MIQP/Exchange 残留”最直接影响审稿人对一审修订是否彻底的判断。其余 Reviewer 已解决项应予以肯定，不需要重做整篇 Response 结构。

---

# 二、修改后的内容是否逻辑自洽、证明是否有效

## 2.1 整体逻辑与证明边界

SCSP 主结构仍是：**CSO** 对有限采样接触候选计算基于 SCM 的接触先验，**Ranking** 在上层高效指导与低层可执行邻近接触之间进行协调，**CPO** 生成 approach / contact / detach / re-contact 轨迹。这条设计链条仍然合理。

但以下命题需要区分：

- SCM 的 **closed-form expression** 是近似/替代接触模型的表达式，并非一般耦合 LCP 的精确闭式解。
- 固定接触集合不能逻辑推出所有间隙 `phi≡0`；需要另外的几何假设。
- `softplus` 在固定参数下光滑，不代表附带 `max`、`clip` 和接触模式切换的最终状态预测映射处处光滑。
- 有限候选点的优化“最佳选择”不能等同于连续物体表面的全局最优，CEM 也不自动提供一般全局收敛保证。
- 经验成功率和 SCM 速度误差实验能支持**已测试场景**中的效果，不能替代理论上对任意几何和接触模式的保证。

### 必须理解的三个核心公式问题

**（a）Eq.(29) 与 Eq.(31) 的循环定义（M03）**

现稿同时写

$$\hat{\beta}_{env}=\alpha\beta_{env}^{(0)},\qquad s=-D\hat{\beta}_{env},\qquad \hat{\beta}_{env,i}=\max(-D^{-1}s_i,0).$$

后二者代入变为 $\hat\beta=\max(\hat\beta,0)$，仅声明非负性，不能给出可执行的预测模型。应统一 residual、初始冲量、投影系数和最终冲量的计算次序。

**（b）Eq.(27) 的 Frobenius 声明与 Appendix A（M01/M20）**

对于任意矩阵 $W$，无约束对角矩阵的最小 Frobenius 误差解是 $\operatorname{diag}(W)$；加入 $\epsilon I$ 一般不保留同一无约束最优性。附录求的是未正则化矩阵 $A$ 的对角部分，必须清楚区分“最佳对角近似”和“为数值稳定另外添加正则”。

**（c）CPO 目标 Eq.(35)/(46)（M08/M12）**

Eq.(35) 混用 $i/k$ 与 $N/T$；Eq.(46) 文字将终端代价称为 $V_f$，公式却写成 $w_{cso}\ell_{cso}$。如果此项是 CSO 已输出的常数，它不会影响 CPO 最小化；只有明确定义为依赖 CPO 预测状态（如 $\ell_{cso}(x_N)$）或其它决策变量时才起作用。应按实际代码统一表达。

## 2.2 数学、实验和统计逻辑的逐点批注

### [M01] P7｜P0｜数学论断与证明不符

**定位**：`closest diagonal approximation`。

**问题**：正文 Eq.(27) 为 D=diag(W_ii)+epsilon I，并称其为 W_ii 的 Frobenius 最优对角近似；但无约束最优解是 diag(W_ii)。附录证明的对象是 A_ii=W_ii-R_ii。

**建议**：改为“对角部分的最佳近似 + 为稳定性额外增加正则项”；必要时将 Eq.(27)、Appendix A 的 A/W/R/epsilon 定义完全对齐。

### [M02] P7｜P0｜Lemma 2 证明假设不足

**定位**：`Based on Assumption 1`。

**问题**：Lemma 2 用“固定接触集合 ⇒ phi≡0”，并将接触 Jacobian 视作常数。固定 contact set 本身并不保证零间隙或冻结几何。

**建议**：增列零间隙/接触几何冻结等假设，限制结论为固定局部线性化条件下成立；说明非零渗透/接触变化时不适用。

### [M03] P7｜P0｜Eq. (31) 循环定义

**定位**：`we define s =`。

**问题**：正文设 s=-D beta_hat_env，Eq.(31) 又给 beta_hat_env,i=max(-D^{-1}s_i,0)。直接代入仅得到 beta_hat=max(beta_hat,0)，不能独立计算 beta_hat，且和 Eq.(29) alpha 投影未清楚衔接。

**建议**：用可计算的初始 residual/impulse 量定义 s，并将 Eq.(29) 作为最终 beta_hat 过程；明确 beta^(0)、beta_hat、softplus/max 的先后关系。

### [M04] P7｜P1｜新增投影的梯度条件需统一

**定位**：`To partially account for the inter-contact coupling`。

**问题**：Eq.(29) 的 beta_hat=alpha beta^(0) 部分重新引入全局耦合。其梯度不仅包含 beta^(0) 的导数，也包含 alpha 的导数，且 clip 分支不可处处微分。

**建议**：说明最终 SCM 是否采用全投影导数，或仅使用 CEM 无梯度评价；相应修订 Lemma 2/Eq.(34) 的 smooth 或 gradient-based 表述。

### [M05] P7｜P1｜力-冲量量纲需检查

**定位**：`T = 10−4`。

**问题**：Eq.(26)/(28) 用 softplus 近似力，后文 Eq.(29) 以 N·s 设置冲量平滑温度 T；若同一 T 对应两种不同量纲，则数学规格不一致。

**建议**：将 force-level 与 impulse-level 的 T 分开命名或给出明确的 h 换算；同时核对 Eq.(19)/(30) 中速度增量和 1/h 的定义。

### [M06] P7｜P1｜softplus / hard-max / projection 流程有歧义

**定位**：`For the CSO formulation below`。

**问题**：文中先使用 softplus，再在 Eq.(31) 使用 hard max；无法判断内层优化实际上调用了哪一种 surrogate 与投影版本。

**建议**：写出按时间顺序的 SCM 伪代码或以一条统一的连续状态映射覆盖 Eq.(26)–(31)，并标出各自使用场景。

### [M07] P8｜P1｜Eq. (34) 梯度推导不完整

**定位**：`The impulse derivative follows`。

**问题**：Eq.(34) 说 impulse derivative follows from Eq.(28)，但当前最终状态传播使用 Eq.(29) 的 alpha beta^(0)，还含 clip/max 分支。

**建议**：依据最终状态传播模型重新定义 ∂beta_hat/∂lambda_r；若仅用于理论讨论，说明实际 GPU CEM 并不依赖该梯度。

### [M08] P8｜P1｜Eq.(35) 时间索引冲突

**定位**：`single shooting problem`。

**问题**：Eq.(35) 求和用 i、内部代价用 k，且目标端点写 N、约束上界写 T，难以确定 CPO horizon 及终端时刻。

**建议**：统一为 k=0,…,N-1，状态 x_N 与 Vf(x_N)，并让全文同一符号代表同一时域。

### [M09] P9｜P1｜Algorithm 2 与 Eq.(45) 不一致

**定位**：`Algorithm 2 Ranking Strategy`。

**问题**：Algorithm 2 按 γ<1/2 硬切换，令 ℓ_lp=ℓ_lift 或 ℓ_place；Eq.(45) 却定义 ℓ_lp=(1-γ)ℓ_lift+γℓ_place 连续混合。两者对 γ∈(0,1) 不等价。

**建议**：明确 Algorithm 2 只选择 reference point / mode label，还是使用 Eq.(45) 连续权重；同步伪代码与实际实现。

### [M10] P9｜P2｜Ranking score 不等于物理可行性

**定位**：`sufficiently competitive cost reduction`。

**问题**：ρ 归一化预测收益只能反映候选相对收益，不能单独证明摩擦可行、IK 可达或能成功接触；文中部分措辞将 competitive 与 feasible 混同。

**建议**：将判断名称改为 quality/suitability，并区分 valid-region 几何筛选、力约束与真正执行可行性。

### [M11] P10｜P1｜CPO 融合代价与模式定义需一致

**定位**：`The overall lift-and-place cost`。

**问题**：Eq.(45) 将 lifting 和 placing 连续加权，但 Algorithm 2 对同一个 ℓ_lp 输出了离散代价，可能改变梯度和行为。

**建议**：根据代码明确 blended cost 与 binary mode 的关系，绘制或文字说明 γ 对两个代价的具体权重。

### [M12] P10｜P0｜终端代价未明确定义

**定位**：`By defining the terminal cost`。

**问题**：正文声称 Vf 与 Eq.(7) 相同，但 Eq.(46) 表现为 w_cso ℓ_cso。若该项是 CSO 已评价的固定标量，它对 CPO 的控制优化不起作用。

**建议**：显式写成 Vf(x_N) 或 ℓ_cso(x_N) 并核对真正运行的目标函数；统一 Eq.(35)/(46) 的 N/T、i/k 和末端状态。

### [M13] P10｜P2｜“避障/无碰撞”推断强于势场定义

**定位**：`The lifting cost is formulated`。

**问题**：提升排斥势基于末端到物体中心的距离，对不规则/非凸物体无法单独保证整个绕行动作 collision-free。

**建议**：若有碰撞检测或几何约束请明确；否则将“guarantee collision-free”类措辞降为 heuristic detouring/repulsion。

### [M14] P13｜P2｜两类成功率指标口径不清

**定位**：`Across 70 paired trials`。

**问题**：正文 direct-force paired trials 报 38.6%，Table V Ours 的 Success(real) 为 0.88；两者可能测的是不同评测，但没有清楚列出分母/仿真与物理条件。

**建议**：分别定义 ideal / real / direct-force 的实验输入、评估标准、样本量与成功率分母，避免被误读成自相矛盾。

### [M15] P13｜P2｜反事实估计仍缺统计细节

**定位**：`Counterfactual Analysis`。

**问题**：新反事实分析有针对性，但缺少独立起点数量、每状态复现实验次数、均值不确定性和保持反事实动作直到接触建立所造成的偏差说明。

**建议**：说明状态选择/随机配对、回滚保护规则与样本量；给出置信区间或标准误，限制结论为该评估协议下的经验收益。

### [M16] P14｜P2｜50 Hz 与 planning-time 口径待区分

**定位**：`fSCSP`。

**问题**：Table VI 给 50 Hz 更新，而 Table II 报 22.91–24.12 ms 的平均 planning time。可能是异步或不同实验口径，但文中未统一说明。

**建议**：标明控制环/规划环/统计环境，是否异步执行、P50/P95 延迟，明确哪项支持实时性。

### [M17] P15｜P2｜置信区间表述的统计含义

**定位**：`highest lower and upper bounds`。

**问题**：即使 SCSP 的 Wilson CI 上下界在各行最大，也不能仅据此得出所有对比均具有统计显著性。

**建议**：可保留“higher estimated success and confidence bounds”，不要由此额外宣称显著性；若要比较差异应做配对或两比例检验。

### [M18] P16｜P0｜Table VII 数据明显可疑，需原始日志

**定位**：`19.25 ± 9.58`。

**问题**：Table VII 中五种 SCSP 条件 Smooth 与 Rough 的 ET 均值和标准差全部完全相同，而部分成功率不同；这是强烈的数据/制表疑点，但不能凭 PDF 断定数据作假。

**建议**：逐组重算原始日志，明确 ET 只计成功还是全部有效 trial；若只是复制错误，更新两行并检查其他统计项。

### [M19] P16｜P2｜失败事件/检测事件的分母需解释

**定位**：`300/300`。

**问题**：Table VIII 的 Contact Estimation 一行大量出现 300/300，但该表同时称 failure-case analysis；可能是事件发生频率而非终止失败次数，语义需要明确。

**建议**：在表下注明“每次观察到事件”的精确定义、事件是否可重叠、recovery rate 的分母及何时认定失败。

### [M20] P20｜P0｜Appendix A 未证明 Eq.(27) 的最优性

**定位**：`Equation (27) adds`。

**问题**：Appendix A 证明 diag(A_ii) 最优，之后称 Eq.(27) 额外加 rI；但 Eq.(27) 已是 diag(W_ii)+epsilon I。若 W_ii 包含 R_ii，则正则量可能重复，且加正则不保留无约束 Frobenius 最优性质。

**建议**：明确 A_ii/W_ii/R_ii/epsilon 的关系。准确陈述“未正则化的最佳对角近似，另加稳定性正则项”，勿称增加后的 D 仍无约束最优。

### 第二类结论

**数学上不能直接认定“新增所有证明已经有效”**：M01、M02、M03、M12、M20 需要优先修订；M04–M07、M08–M11 涉及最终梯度与目标的定义精确性。**数据上不能将疑点直接认定为造假或结果错误**：M14、M16、M18、M19 均需回到实验日志/统计代码核实。尤其 Fig.10 属于可以直接从 PDF 确认的图表重复错误，已列第一类 R08。

---

# 三、修改稿语言风格是否统一，与原文是否差异较大

## 3.1 风格评价及比较边界

总体上，保留文字与红色新增段落仍属于一致的 IEEE T-RO 机器人模型预测控制论文风格：**问题提出—方法定义—求解器—实验验证—局限性**。新增内容更强调理论限制和模块归因，这符合 Reviewer 要求，并不构成明显文风断裂。无需为了统一语气而全文重写。

由于此轮可访问材料只有修改稿及审稿文本，没有取得单独的未修订原始 arXiv PDF，**不能把红黑修订色比较冒充为完整的原文逐句差异审计**。可以肯定的是：新增的解释性段落比部分保留文本更谨慎，但个别技术术语仍有过度外推。

真正需要的不是一般英文润色，而是：

1. **术语**：`sampled-set optimal / global / feasible / smooth / collision-free` 必须有准确边界；不能将相对收益高的 contact 直接称为物理 feasible。
2. **命名**：`MIQP`、`GPU-parallel CEM`、`Sampling and Exchange`、`Ranking Strategy` 等必须与图、算法和时间表一致。
3. **图文对应**：Fig.2 的旧公式编号和拼写、Fig.10 的双图、Table IX 单位均需校对。
4. **语法和符号**：`α`、`N/T`、`i/k`、`p_ref`、`β/λ` 的局部混用应从源头修改，而不是仅在翻译层面润色。

## 3.2 需要在 PDF 直接标注的语言与版面问题

### [L01] P5｜P2｜图内拼写与旧编号

**定位**：`Ranking Stragegy`。

**问题**：Fig.2 中 Ranking Stragegy 拼写错误，同时 Eq.(32)/(38)/(40)/(47) 可能沿用旧稿编号，应与现有公式逐项交叉核对。

**建议**：改正为 Ranking Strategy；统一图内所有公式索引，并同步图注及 Algorithm 2。

### [L02] P8｜P2｜过强/易歧义的术语

**定位**：`as a pointwise global search`。

**问题**：“pointwise global search” 容易被理解为连续接触表面上的全局最优，与已承认的有限采样搜索不完全一致。

**建议**：统一改写为“search over the finite sampled contact candidates”，或加限定“global only within the sampled set”。

### [L03] P12｜P2｜参数扫描无法证明一般最优值

**定位**：`There exists an optimal convergence value`。

**问题**：Table III 仅测试有限候选点数量，不能据此断言一般性的 optimal convergence value；这也是 reviewer 对理论表述强度的敏感点。

**建议**：改为“performance plateaus near 70–120 samples in our tested settings”。

### [L04] P17｜P2｜Table IX 单位混杂

**定位**：`RS cost time`。

**问题**：Table IX 的 RS cost time 为 (s)，而 CSO、CPO 其他主要计时项为 (ms)；值 0.01s 可能被误读为 0.01ms。

**建议**：尽可能统一为毫秒；若保留秒，请明确换算且核对 Total planning time 的求和口径。

### [L05] P19｜P2｜PointWorld 类比需收窄

**定位**：`model in [52]`。

**问题**：Reviewer 5 不希望 CSO/SCM 被概念扩大为 world model。此处建议 PointWorld 作为可能的 future model 不等于已经可直接替换 SCSP SCM。

**建议**：明确 ContactSDF/PointWorld 的接口、任务域差异及“potential direction rather than drop-in replacement”。

### 第三类结论

**语言风格总体统一，且与保留的原文写法差异不大**。优先修技术逻辑与实验图表，再进行一次专门的 IEEE 英文终校；避免先做大规模重写，让修改稿失去原有章节结构。

---

# 四、修改稿是否破坏原文的技术逻辑

## 4.1 保留下来的核心技术链

```text
Current object/robot observation + desired object goal
        ↓
CSO: offline surface sampling; online valid-region filtering;
     SCM-based candidate contact/force evaluation
        ↓
Ranking: compare CSO-guided candidate and nearby candidate;
         smooth accept/switch decision
        ↓
CPO: prior-guided lifting, placing, contact, and replanning
        ↓
Robot execution and closed-loop state feedback
```

**结论**：这个级联结构没有被修改破坏，反而因为 placing cost 显式使用 $p_{ref}$ 而更符合高层 guidance 应影响低层动作的目的。旧版中存在的接触点参考未进入 placing cost 的错误，修改为 $p_{ref}-\delta n_{ref}$（$\delta=0.02$ m）是有技术动机的修复，不能把这一修改本身说成“破坏技术逻辑”。

## 4.2 需要区分“改文稿”还是“改算法”的变动

| 子模块 | 新稿与审稿所指旧稿之间的变化 | 可能的实际影响 | 应核对的证据 |
|---|---|---|---|
| SCM | softplus、初始冲量、全局 $\alpha$ 投影 | 改变环境冲量和预测动力学 | Eq.(26)–(31) 与生成 Table IV/V 的代码 |
| CSO Solver | 从旧稿 MIQP/SNOPT 说法修正为枚举+CEM | 计算预算、梯度性质、最优性表述不同 | Solver 调用、参数、Table IX 测时 |
| Ranking | 旧 T1–T3 计数器/迟滞 → EMA 平滑 | 改变响应滞后、切换频率和接触保持行为 | Algorithm 2、Table II/VI、Fig.10 的版本 |
| Lifting/placing | 排斥势改符号；placing 明确 $p_{ref}$ 与向内 2 cm | 改变梯度方向和接触目标位置 | 势场代码及 ablation A8/A9 |
| Success threshold | 原审稿意见暗示约 26°，新版说明 14.1° | 如果实际变更阈值，就改变成功率定义 | 评估脚本、历史和新版实验结果 |
| 统计试验 | 硬件每条件从少量 trial 扩大到 30 | 结果更可靠，但不应照抄旧 ET | 完整原始日志和 Wilson CI 统计脚本 |

实际是否“改变算法”，不能单靠新版 PDF 判定：有可能是**旧稿写错而代码一直正确**，也可能是**本轮真的修改了代码**。审查结论必须以代码版本和实验记录为准。

## 4.3 需要在 PDF 核实的实现保真问题

### [T01] P11｜核实｜成功阈值可能发生实质变化

**定位**：`orientation tolerance corresponding`。

**问题**：Reviewer 3 认为旧稿的四元数阈值约等于26°；修改稿用 0.015 并说明14.1°。如果不是旧稿排版误植而是实际改变阈值，就需要相同口径重算所有成功率。

**建议**：对照原代码和原文的 success threshold，说明这是纯纠错还是实验协议改动；若改动，重算/重新报告受影响比较。

### [T02] P11｜核实｜论文宣称与实际启用模块的关系

**定位**：`repulsive-field terms are disabled`。

**问题**：主实验注明 object-center attraction 和 repulsive-field terms 被关闭，但论文将 repulsion 描述为主要 lifting 规划能力之一；需要说明哪些任务实际启用、哪些是通用公式。

**建议**：按任务列出启用的损失项及权重，明确当前实验性能由哪些子项贡献，避免把关闭模块归因于性能提升。

### [T03] P9｜核实｜迟滞计数器改为 EMA 可能改变行为

**定位**：`γt+1`。

**问题**：旧稿的 T1–T3 计数/迟滞机制被 EMA 替换，不是单纯语言修正；在 β=0.2 时，持续 κ=1 的四次更新可使 γ 从0跨过0.5阈值。

**建议**：核实 Table II/VI、Fig.10 和硬件实验是否都运行当前 EMA 机制；必要时标明算法实质更新并重复评测。

### [T04] P8｜核实｜求解器实现与实验版本需一致

**定位**：`GPU-parallel CEM`。

**问题**：修改稿改用 GPU CEM 描述力优化/候选枚举；若原实验实际运行 SNOPT 或别的流程，当前算法描述与结果将不匹配。

**建议**：用同一版本代码核对 Fig.2、Algorithm1、Table II/V/IX 的 solver 配置与计时，明确改的是写法还是实现。

## 4.4 代码与实验的最小核对表

| 对象 | 必须确认的对应关系 |
|---|---|
| SCM Eq.(26)–(31) | 数学定义、实际计算顺序、投影系数、软硬截断是否与代码一致 |
| CEM / CSO | P8 Solver 描述与 Fig.2、Algorithm 1、Table IX、Table II/V 完全一致 |
| Ranking EMA | Fig.10、Table VI、主消融和硬件 trial 均采用相同控制机制 |
| CPO Eq.(44)–(46) | $p_{ref}$ 目标、控制权重、terminal objective 与代码一致 |
| Success criterion | 四元数阈值、位置阈值、稳定时间和 trial 结束条件在各方法中一致 |
| Hardware Table VII | Smooth/Rough ET 是否从各自日志正确计算；失败/成功时间是否同一口径 |
| Timing | 50 Hz 更新频率与平均规划耗时的不同平台/并行口径有说明 |

## 4.5 综合修订优先级（与四个大类一一对应）

| 顺序 | 问题 / PDF 评论 ID | 核心理由 |
|---|---|---|
| **P0-1** | SCM 定义 M03；Appendix A 的 M01/M20 | 关键数学定义/证明尚未正确闭合 |
| **P0-2** | Fig.10 R08 | 直接影响 Reviewer 5 和 Editorial Board 的核心新增证据 |
| **P0-3** | Table VII 的 ET M18 | 五组 Smooth/Rough 均值和标准差全重复，必须核对原始数据 |
| **P0-4** | CPO Eq.(46) M12 | 终端代价对优化变量是否有效不清楚 |
| **P1-1** | Ranking M09/M11/T03 | 离散伪代码与连续目标是否对应，以及 EMA 是否应用于实验 |
| **P1-2** | MIQP/CEM/Exchange R01/R02/R12/R13/T04 | 原审稿人反复关注的真实实现和可复现性 |
| **P1-3** | SCM Lemma 2 / 梯度 M02/M04–M07 | 假设、平滑性、单位和最终梯度应与实际模型一致 |
| **P1-4** | 泛化主张 R09/R10/R11 | 不应再对未证明的一般多接触/奇异规避做强断言 |
| **核实** | T01、T02、M14、M16、M19 | 需要原始阈值、代码或日志，不能仅凭文稿定性 |
| **P2** | R03–R07、L01–L05 及剩余备注 | 参数、术语、图号、排版与文字论断精修 |

### 最终审稿判断

1. **一、回应完整性**：所有主要 Reviewer 的议题基本都有针对性处理；但 MIQP/CEM/Exchange、Fig.10 等关键问题还不够完整。**不应否认本轮修订的实质工作量**。
2. **二、逻辑与证明**：**总体技术主线合理，但并非所有新增推导都有效**。SCM 的 Eq.(27)/(31)、Lemma 2、Appendix A、CPO Eq.(46) 应优先修订。
3. **三、语言风格**：**基本统一**。精修术语、章节图号、数据口径和论断强度即可；完整原文与修订稿逐句对比仍需原版 PDF。
4. **四、原技术逻辑是否受损**：**核心 CSO→Ranking→CPO 设计没有被破坏**；但部分底层算法描述已实质变化，必须核对方法、代码和实验是否一致。

**建议提交前的操作顺序**：先核对实现代码与实验日志 → 修复数学定义与关键实验图 → 统一全篇 Solver/Ranking/CPO 名称与目标 → 最后做语言及排版校对。这样不需要推倒重写 SCSP，但能重点消除二审最可能再次质疑的地方。
