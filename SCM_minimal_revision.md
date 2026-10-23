# SCM 单步耦合残差修正：最小修改清单

**原则：**保留对角 Delassus 近似、Softplus、接触集合假设、CSO/CPO 框架；仅将式 (29) 的全局 `clip` 投影替换为一次光滑的耦合残差修正。**这是一项待验证的模型修改，不能沿用旧模型的实验数值作为新模型结果。**

## 1. 第 7 页，Section III-C，式 (29)（核心替换）

保留式 (25)–(28)，将式 (26)/(28) 说明为**初始对角近似**。把式 (29) 整体替换为：

$$
\begin{aligned}
\boldsymbol\beta_{\mathrm{env}}^{(0)}
&=\operatorname{softplus}_{T}(-D^{-1}\boldsymbol g_{\mathrm{env}}^{\mathrm{step}}),\\
\boldsymbol r_{\mathrm{env}}
&=W_{\mathrm{env}}\boldsymbol\beta_{\mathrm{env}}^{(0)}
  +\boldsymbol g_{\mathrm{env}}^{\mathrm{step}},\\
\hat{\boldsymbol\beta}_{\mathrm{env}}
&=\operatorname{softplus}_{T}\!\left(
\boldsymbol\beta_{\mathrm{env}}^{(0)}-\frac{\boldsymbol r_{\mathrm{env}}}{L}
\right),\\
L&=\lambda_{\max}(W_{\mathrm{env}}).
\end{aligned}
$$

其中 $W_{\mathrm{env}}$ 取式 (22) 的**完整耦合矩阵**（而非式 (23) 的块对角近似）。保留原文 $\boldsymbol g_{\mathrm{env}}^{\mathrm{step}}$、$d_{\mathrm{env}}$ 和 $T$ 的定义。

**替换式 (29) 前的英文说明：**

> To partially compensate for the neglected inter-contact coupling, we apply a single smooth residual correction to the diagonal impulse estimate. This avoids the hard clipping and additional inner iterations of the previous projection-based formulation.

**删除式 (29) 后与旧投影有关的文字：**共享缩放系数 $\alpha$、`clip`、$\alpha=1$ 的边界处理，以及“one projection per prediction step”等说明。

**补充一句实现说明：**

> The product $W_{\mathrm{env}}\boldsymbol\beta$ is evaluated as $\tilde J_{\mathrm{env}}M_o^{-1}(\tilde J_{\mathrm{env}}^\top\boldsymbol\beta)+\epsilon\boldsymbol\beta$, without forming the full Delassus matrix. For fixed contact geometry, $L$ is computed once per CSO update and reused during force optimization.

## 2. 第 7 页，式 (30)–(31)

- **式 (30) 不改**：直接使用新式 (29) 输出的 $\hat{\boldsymbol\beta}_{\mathrm{env}}$。
- **式 (31) 删除**旧约束 $\hat\beta_{\mathrm{env},i}=\max(-D^{-1}s_i,0)$；因为环境冲量已由式 (29) 和 $f_{\mathrm{SCM}}$ 定义。
- 删除紧随式 (31) 的 “Here $s_i$ is ... defined in (29)” 句子；无需再定义 $s=-D\hat\beta$。

## 3. 第 7–8 页，少量文字同步

- **Lemma 2（式 (28)）**：将“最终环境接触力是式 (28) 的光滑闭式函数”限定为“**初始对角接触力估计**是光滑闭式函数”；最终冲量按式 (29) 得到。
- **第 8 页，式 (34) 后**：把 “The impulse derivative follows from (28) ... with ... clipping branches. The resulting map is piecewise smooth” 替换为：

> The impulse derivative follows from (29). Under fixed contact geometry, the residual correction and Softplus operations are smooth with respect to the robot contact forces. The continuous force subproblem is solved for each candidate using GPU-parallel CEM, and the candidate costs are enumerated.

- **Fig. 4**：图中若画有全局投影或 `clip`，改为 “Single Residual Correction”；其余示意结构保留。

## 4. 实验（必须核实，其他正文尽量不动）

新模型的数值结果尚未验证，**至少重新计算 Table IV 的 LCP 预测误差和 Table V 的 SCM 性能**；若将新模型用于全部 SCSP 实验，其他受影响的表格和结论也要按实际结果更新。建议附加一个有/无修正的消融。

**实现提醒：**$L$ 固定后，每次 CSO 目标函数评估仍需重新计算一次 $W\beta^{(0)}$；只是**不需要重新计算最大特征值，也不需要额外迭代优化**。该方案平滑、闭式且无新增经验增益，但不能预先保证具有旧 Projection 的预测精度。
