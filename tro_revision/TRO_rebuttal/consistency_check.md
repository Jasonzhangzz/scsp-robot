检查日期：2026-10-26。

已清理 `TRO_rebuttal/root.tex` 中的删除标记及其内容，删除整张被横线覆盖的旧 SCM 误差表，保留新版表格，并将回复正文和回复表格中的红、蓝修订文字恢复为黑色。审稿人标题和审稿意见保留原有蓝色。删除宏、添加宏及替换宏已从回复信源码中展开或移除，旧表的残留引用也已清理。

以下检查覆盖 Reviewer 1 的 9 条、Reviewer 3 的 5 条、Reviewer 5 的 4 条，共 18 条回复。以当前 `tro_revision/root.tex` 实际包含的正文为准，重点核查 `method.tex` 和 `experiment.tex`；涉及引言、预备知识、泛化实验和局限性时，也核查了主文件实际包含的 `introduction.tex`、`preliminary.tex`、`generalize.tex` 和 `discussion.tex`。公式、图表及文献编号来自当前源码重新编译的结果，没有沿用旧 PDF 的编号。

本次对技术内容只做核查，未擅自决定有争议的目标函数、修改实验数据或改写实质性回复。下列“部分匹配”区分了正文与回复的直接差异，以及回复尚未充分回答审稿要求的情况。

| 回复 | 核查结论 | 正文依据及待处理事项 |
| --- | --- | --- |
| R1 Q1：MIQP/QP 定性 | 匹配 | 方法 III-C/D 采用三轮 Gauss--Seidel SCM 和有限候选集上的连续力优化；实验明确 GPU 并行 CEM；表 IX 已移除 MIQP 计时行。Eq. (27)、表 V、表 IX 编号正确。 |
| R1 Q2：SCM 假设、误差界及验证 | 部分匹配 | 误差界已移除，表 IV 的全部 44 组均值/标准差与回复中的表格逐项相同；模型、参数、表 V 成功率均匹配。但“所有接触状态均为厘米每秒级”的概括不符数值，初始状态数量也有歧义。见 A2、A3。 |
| R1 Q3：排名策略 | 部分匹配 | 归一化方向、二值触发、EMA 和 Eq. (36) 均匹配。但回复声称按包含力正则项的完整候选目标排名，正文没有这样定义。见 A1。 |
| R1 Q4：CPO 势函数 | 匹配 | 方法 IV-C 的排斥项已改为负对数，放置目标为向内偏移 0.02 m 的表面点，旧物体中心吸引项已移除。实验中排斥项关闭，R3 Q4 的回复对此也有说明。 |
| R1 Q5：各组件贡献 | 部分匹配 | A1--A7、A9、表 IV--VI 和硬件事件表确实存在。但表 VIII 没有回复所说的“接触切换后倾倒及恢复”专项统计；独立 random/multi-start、代价与成功率相关性以及分项 lift/place 消融仍未提供。见 A4、B1。 |
| R1 Q6：基线与对比协议 | 部分匹配 | Sampling 原始 C3/Gurobi 配置、表 I 列名、STOCS-3D、ContactSDF、A5、A7 参数及成功率匹配。但 A7 的物体代价符号与正文不同，且“排除了求解器差异”的因果结论不成立；A5 也没有采用同等时间预算。见 A1、A5。 |
| R1 Q7：多接触与泛化范围 | 主要匹配 | VI-A(3) 确有 top-5 GWB 区域对、指尖附近候选点及 Eq. (47) 的有限扰动力残差；VII 确有组合增长和适用范围限制。回复中的“有限采样不认证六维力闭合”限定尚未明确写入正文，建议同步。见 B2。 |
| R1 Q8：硬件统计 | 匹配，试验总数建议澄清 | 表 VII 为每方法、每任务/表面条件 30 次；五种任务、两种表面对应每方法 300 次。Wilson 区间和表 VIII 的五类事件及恢复率均存在。“300 real-world trials in total”宜明确为每方法，避免被理解为含全部比较方法的总数。 |
| R1 Q9：技术修正 | 匹配 | Eq. (29) 摩擦约束含双侧切向界；Eq. (27)--(28) 使用力单位和相应 h 因子；最优接触点符号、无量纲四元数误差、固定接触条件下的平滑性描述与正文一致。旧 Lemma 4 已移除。 |
| R3 Q1：排名定义冲突 | 部分匹配 | EMA 初值、阈值、相位方向、Algorithm 2 和 Fig. 6 均匹配；完整目标及力正则项的排名定义与正文有 A1 所述差异。 |
| R3 Q2：优化声明与 Exchange | 匹配，最优性措辞可收紧 | 三轮扫描公式与正文 Eq. (27) 一致；CEM 和 Exchange 掩码步骤也一致。有限候选集的“最优”仍应理解为有限次 CEM 获得的最低已求得代价，正文和回复均没有连续子问题全局最优的保证。 |
| R3 Q3：文字、表格及置信区间 | 主要匹配，回复尚不完整 | 硬件旋转已改为 119/120，表 VII 为百分比及 Wilson 区间；表 X 含开环结果，多指抓取已改为 slight advantage。但审稿人同时建议的模拟试验置信区间仍未提供。见 B3。 |
| R3 Q4：可复现参数 | 匹配 | 摩擦界、吸引权重 20、关闭排斥项、两项路径余量、70 个候选、20 轮 CEM、acados/HPIPM 参数、控制界和离线 FPS 成本均可在正文找到。 |
| R3 Q5：小修正 | 部分匹配 | Q 矩阵、终端索引、切向基退化处理、14.1 度容差及离散质量分布匹配。但回复给出的完整目标没有写入 Eq. (29)，CPO 终端项也使用了不同的代价符号。见 A1。 |
| R5 Q1：排名价值、模型限制与校准 | 部分匹配 | 200 条轨迹、每条 10 个决策点、折扣 0.99、归一化优势和一次性校准均匹配。回复漏述正文中的反事实决策保护协议；接触点变化频率也不等于排名触发频率；“误差不显著影响决策”超出了现有验证。见 A4、A6。 |
| R5 Q2：清晰性、掩码、时间及案例 | 部分匹配 | 掩码实例、离线分割、案例新标题、Fig. 20、20/20 对 15/20 和 VII 的局限性讨论匹配。但“这些启发式并非实验必要组件”与已使用的掩码描述不协调；横轴 step 与实际时间的对应仍未交代完整。见 A7、A8。图中文字可读性是否优于上一版，不能仅凭当前 TeX 判断。 |
| R5 Q3：SCM 理论意义 | 主要匹配，结论应限定 | 三轮扫描、移除误差界、表 IV/V 及部署成功率均匹配。“误差处于可接受范围”和“扫描纠正偏差”尚无明确精度阈值或 K 的对照实验，应作为有限任务下的经验判断。见 A2、A6。 |
| R5 Q4：拼写与补充说明 | 匹配 | 被指出的旧句、LCS 和孤立 I-MPC 已不在当前正文；相关工作分别引用现 [31]、[32]；Eq. (6) 索引为 k，FPS 成本明确离线；h、方向容差和质量分布均匹配。 |

建议优先处理以下差异。

1. **A1：目标函数与排名依据没有统一。** 回复 [R1 Q3](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:138)、[R3 Q1](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:300)、[R3 Q5](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:387) 明确写出 `J = ell_cso + ell_reg`，并声称排名使用这个完整目标。正文 [Eq. (29) 及其解释](/home/lab423/scsp/scsp-robot/tro_revision/method.tex:284) 却只有 `ell_pose(x_T)`，还明确说 CSO 的目标就是姿态代价；[排名定义](/home/lab423/scsp/scsp-robot/tro_revision/method.tex:362) 使用 `ell(x_t) - ell(x_hat_t^+)`，没有 J 或正则项。[实验设置](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:22) 又给出了 0.01 和 0.03 两个力正则权重，因此正文自身也不统一。此外，回复 R1 Q6、R3 Q5 使用 `ell_cso`，而正文 [A7](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:116) 和 [CPO 终端代价](/home/lab423/scsp/scsp-robot/tro_revision/method.tex:477) 使用 `ell_pose`。应根据实际实现决定：若排名确实使用完整目标，就补齐 Eq. (29)、J、正则项和排名定义；若排名只使用姿态改善，就修改回复。现有文件不足以判断实现究竟采用哪一种。

2. **A2：SCM 误差的概括超出了表格支持。** [R1 Q2](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:110) 写“linear errors ... centimetre-per-second level in all contact states”，但表 IV 中 IsaacGym 接触转换的线速度均值为 0.42、0.37、0.37 m/s；翻转 rollout 为 0.14 ± 0.75 m/s，角速度为 9.73 ± 37.64 rad/s。建议分别说明接触模式和 rollout 分布，并明确 IsaacGym 转换状态及 rollout 的较大误差。[正文的误差总结](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:134) 已排除了转换状态，但“otherwise about 1 rad/s”若包括 rollout，仍不准确。表格本身的 44 组均值/标准差在两份文件中完全相同，问题在概括语句。

3. **A3：SCM 接触转换样本量不明确。** 回复 [R1 Q2](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:82) 和 [R1 Q5](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:195) 均称每对象、每接触状态 32 个初始状态。[正文](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:134) 同时写每对象包含 32 个 impact 和 16 个 lift-off 初始状态，若合并为 transition 就是 48 个；后文又说每状态 32 个。需说明两类是否合并、是否另外抽样、计数在哪一级进行，并同步回复。两份文件重复同一句话不能消除这处正文内部歧义。

4. **A4：接触切换后的恢复统计并未报告，频率名称也不应混用。** [R1 Q5](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:197) 声称表 VIII 报告“切换接触后物体倾倒”的发生次数及恢复率。但 [表 VIII](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:327) 只有 perception、contact estimation、slippage、control saturation 和 poor contact selection，没有“发生在切换之后”或“倾倒之后恢复”的条件统计。正文还在 [A1 分析](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:118) 写过频繁切换导致部分任务失败，因此回复中“Contact switching itself does not directly cause failure”也过于绝对。建议将回复改为该表实际统计的五类事件及事件发生后的最终成功率。另，[R5 Q1](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:465) 将最优接触点变化频率和物理接触切换频率解释为排名策略触发频率；两者并不直接统计 kappa/gamma 的变化。表 VI 的 5.51/4.40 Hz 与 0.203/0.195 Hz 应按原定义报告。

5. **A5：A7 不能单独排除求解器差异，A5 也不是同预算对照。** [R1 Q6](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:231) 最后一句声称差距隔离了 contact guidance 而非 optimizer 的影响。但 [默认 CPO](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:17) 使用 acados SQP-RTI/HPIPM，[A7](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:116) 使用 MPPI，两项因素同时改变。建议改成“在相同预测模型、时域和控制界下，移除 guidance 的 MPPI 变体表现较差”，保留因果推断边界，或增加同求解器的有/无 guidance 对照。表 II 中 A5 每轮规划约 612.53/577.81 ms，SCSP 为 12.12/12.91 ms，也不能称为同 wall-clock budget；现有回复没有明确说明这一限制。

6. **A6：反事实评估的关键执行协议和模型误差结论需要限定。** [正文](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:206) 明确在所选接触建立前保护反事实决策，防止立即被后续排名覆盖；[R5 Q1](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:451) 只说反转决策后继续运行 full SCSP，容易被理解为仅扰动一步，应补上保护协议。[同一回复](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:468) 的“single-step errors do not significantly affect ... decisions”没有决策一致性或显著性分析支持；[VII](/home/lab423/scsp/scsp-robot/tro_revision/discussion.tex:9) 则明确将鲁棒性限定于已测任务。建议表述为测试范围内保持了任务表现。R5 Q3 对三轮扫描能纠正偏差的陈述也宜区分设计动机与已验证效果，因为没有 K=0 对 K=3 的误差消融。

7. **A7：掩码在实验中的角色表述冲突。** [R5 Q2 末尾](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:515) 说这些启发式是 extensibility designs 而非实验的 essential components，但同一回复和 [VI 的足球/关节物体设置](/home/lab423/scsp/scsp-robot/tro_revision/generalize.tex:11) 明确使用地面、工作空间和内部表面掩码来约束并触发接触切换。建议改成：非抓取旋转/翻转主实验不依赖额外语义标注；拓展案例确实使用了所列任务掩码。只有实际设置支持时，才能加入前半句。

8. **A8：时间步、控制周期和执行时间仍需区分。** R5 Q2 询问所有横轴 step 的时间单位，当前回复没有直接回答；R5 Q4 仅补充 h=0.002 s。[实验设置](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:15) 给出最大 2500 rollout steps，若每个执行 step 就是 h，总模拟时间上限为 5 s，但表 II 的平均执行时间经常为 15--25 s。可能是墙钟时间包含推理、控制降频或重复执行，正文没有明确说明，不能直接判定数据错误。还需核对 [表 IX](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:357) 中 RS 的单位确实是 s：0.01 s 为 10 ms，加上 CSO 2.35 ms 和 CPO 8.68 ms 约为 21.03 ms，与表 III 的约 12.52 ms 是否采用不同计时范围，应交代清楚。确认口径后，在相关图注和回复中写明 integration step、control/planning step、wall-clock time 的对应关系。

另有以下未完全覆盖或可以补强的项目。

- **B1，R1 Q5 的证据覆盖。** 有限候选集并行搜索并不等价于增加一个独立 multi-start 基线；“随机接触过于不可靠”也没有实验数据支持。现有结果没有直接报告 CSO cost 与执行成功率的相关性、跟踪两类接触点的实际时间占比，或分别关闭 lift/place 项的消融。应承认未完成的对照，并准确说明现有 A1--A9 能支持什么。
- **B2，R1 Q7 的力闭合限定。** [VI-A(3)](/home/lab423/scsp/scsp-robot/tro_revision/generalize.tex:26) 仍说 inner-loop objective “transformed to force closure”，没有显式加入回复中的“有限扰动采样残差不构成六维力闭合认证”。建议在 Eq. (47) 后同步这句限定。
- **B3，R3 Q3 的模拟试验区间。** 表 VII 已有硬件区间，表 II 没有模拟试验成功率区间；当前实验文字也没有清楚给出表 II 每个任务、每种方法的试验数。表 V 的“100 rotation + 100 flipping”是另一组实验，不能自动代替表 II 的样本数。若暂不补区间，回复应明确只新增了硬件区间。
- **B4，R5 Q1 的感知图注。** [硬件设置段落](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:254) 与回复均说 MuJoCo proxy 没有 object collision geometry，主要用于可视化和粗对齐；[Fig. 13 图注](/home/lab423/scsp/scsp-robot/tro_revision/experiment.tex:258) 却仍说 contact estimation 使用 MuJoCo 实现。需要明确代理模型、接触估计及 SCM 各自使用的几何与接口，避免图注让读者以为通过物体碰撞网格取得真实接触。
- **B5，回复信自身的编号和拼写。** [开头致谢](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:39) 使用 R1/R2/R3，但后文审稿人编号为 1/3/5；宜按实际审稿人编号统一。[STOCS 图注](/home/lab423/scsp/scsp-robot/tro_revision/TRO_rebuttal/root.tex:217) 的 `objec` 应为 `object`。这些不影响正文核对，但会影响最终提交观感。

验证结果：两份 root.tex 均已在独立临时目录通过 latexmk/pdflatex 编译；回复信 10 页，论文 21 页。回复信没有未定义引用、重复标签或 overfull box 警告，仅保留 IEEEtran 无 author 的提示及少量 underfull 提示。论文存在既有排版 overfull 提示，本次没有修改论文。回复信全部 10 页已渲染查看，旧删除表格及横线均已消失；文本颜色检查仅剩黑色与审稿意见原有的深蓝色。正文表 IV 与回复表格的 44 组均值/标准差全部一致。Eq. (27)/(28)/(29)/(36)/(40)/(44)/(47)、Fig. 6/9/10/11/20、Table I--X 及文献 [31]/[32] 的对应编号均已核实。
