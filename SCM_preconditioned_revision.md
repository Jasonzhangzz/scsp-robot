# SCM 残差修正：预条件化步长的最小修改

**出发点：**当前 `tro_revision/method.tex` 的式 (29) 已经是"对角初值 + 一步残差修正"：

$$
\boldsymbol r_{\mathrm{env}}=\boldsymbol W_{\mathrm{env}}\boldsymbol\beta^{(0)}_{\mathrm{env}}+\boldsymbol g^{\mathrm{step}}_{\mathrm{env}},\qquad
\hat{\boldsymbol\beta}_{\mathrm{env}}=\operatorname{softplus}_T\!\Big(\boldsymbol\beta^{(0)}_{\mathrm{env}}-\frac{\boldsymbol r_{\mathrm{env}}}{L}\Big),\qquad
L=\lambda_{\max}(\boldsymbol W_{\mathrm{env}}).
$$

它是对偶 QP $\min_{\boldsymbol\beta\ge0}\tfrac12\boldsymbol\beta^\top\boldsymbol W\boldsymbol\beta+\boldsymbol g^\top\boldsymbol\beta$ 的一步投影梯度。问题是 $L$ 需要特征值，且步长不随每行的有效惯量缩放。下面只换步长，其余不动。

---

## 1. 式 (29)：把 $\boldsymbol r/L$ 换成 $\omega\boldsymbol D^{-1}\boldsymbol r$

$$
\boldsymbol r_{\mathrm{env}}=\boldsymbol W_{\mathrm{env}}\boldsymbol\beta^{(0)}_{\mathrm{env}}+\boldsymbol g^{\mathrm{step}}_{\mathrm{env}},\qquad
\hat{\boldsymbol\beta}_{\mathrm{env}}=\operatorname{softplus}_T\!\Big(\boldsymbol\beta^{(0)}_{\mathrm{env}}-\omega\,\boldsymbol D^{-1}\boldsymbol r_{\mathrm{env}}\Big),\qquad
\omega=\frac{1}{\hat L_N},
$$

$$
\hat L_N=\max_i\sum_j\frac{|W_{ij}|}{\sqrt{D_{ii}D_{jj}}}\;\ge\;\lambda_{\max}\!\big(\boldsymbol D^{-1/2}\boldsymbol W_{\mathrm{env}}\boldsymbol D^{-1/2}\big).
$$

- $\boldsymbol D=\operatorname{diag}(\boldsymbol W_{\mathrm{env}})+\epsilon\boldsymbol I$ 沿用式 (27)，$\boldsymbol W_{\mathrm{env}}$ 仍取式 (22) 的完整耦合矩阵。
- $\hat L_N$ 是归一化 Delassus 矩阵的 Gershgorin 上界，只用矩阵元素，不做特征分解。单刚体下 $\boldsymbol W_{\mathrm{env}}=\tilde{\boldsymbol J}\boldsymbol M_o^{-1}\tilde{\boldsymbol J}^\top+\epsilon\boldsymbol I$ 形成代价 $O(6n^2)$。
- 最保守的替代是 $\omega=1/n$（$n$ 为约束行数），同样有保证，可作为备选。

**式 (29) 前的说明句改为：**

> To partially compensate for the neglected inter-contact coupling, we apply one Jacobi-preconditioned projected-gradient step to the diagonal impulse estimate, with a closed-form step size that guarantees descent.

**式 (29) 后的实现句改为：**

> The product $\boldsymbol W_{\mathrm{env}}\boldsymbol\beta$ is evaluated as $\tilde{\boldsymbol J}_{\mathrm{env}}\boldsymbol M_o^{-1}(\tilde{\boldsymbol J}_{\mathrm{env}}^\top\boldsymbol\beta)+\epsilon\boldsymbol\beta$. For fixed contact geometry, $\boldsymbol D$ and $\hat L_N$ are computed once per CSO update and reused during force optimization.

删除 "$L$ is computed once per CSO update" 和 "where $\boldsymbol W_{\mathrm{env}}$ is the full coupled Delassus matrix … rather than its block-diagonal approximation" 中的 $L$ 字样；后半句保留。

---

## 2. 新增一条 lemma（替代已删除的误差界 Lemma 3）

放在式 (29) 之后、式 (30) 之前：

> **Lemma 3.** Let $\boldsymbol\beta^*$ be the solution of the coupled LCP (21) and $f(\boldsymbol\beta)=\tfrac12\boldsymbol\beta^\top\boldsymbol W_{\mathrm{env}}\boldsymbol\beta+\boldsymbol g^{\mathrm{step}\top}_{\mathrm{env}}\boldsymbol\beta$. With the hard projection $\max(\cdot,0)$ in place of $\operatorname{softplus}_T$, the update in (29) satisfies
> $$f(\hat{\boldsymbol\beta}_{\mathrm{env}})\le f(\boldsymbol\beta^{(0)}_{\mathrm{env}}),\qquad \|\hat{\boldsymbol\beta}_{\mathrm{env}}-\boldsymbol\beta^*\|_{\boldsymbol D}\le\|\boldsymbol\beta^{(0)}_{\mathrm{env}}-\boldsymbol\beta^*\|_{\boldsymbol D}.$$
> With $\operatorname{softplus}_T$, both inequalities hold up to an $O(T\sqrt{n})$ perturbation.

**证明要点（可放附录）：**在 $\boldsymbol D$-内积下，$f$ 的梯度 Lipschitz 常数为 $\lambda_{\max}(\boldsymbol D^{-1/2}\boldsymbol W\boldsymbol D^{-1/2})\le\hat L_N$；步长 $\omega\le1/\hat L_N$ 的投影梯度对凸二次目标单调下降，且以 $\boldsymbol\beta^*$ 为不动点、非扩张。

这条不需要 $\lambda_{\min}$，直接回应审稿人对旧界 $\sigma_1/\lambda_{\min}$ 的质疑。

---

## 3. 其余公式同步（只改文字）

- **式 (26)/(28)：**力形式的 $\operatorname{softplus}_T$ 与冲量形式用同一个 $T=10^{-4}\,\mathrm{N\,s}$，量纲不一致。统一写成冲量形式 $\boldsymbol\beta^{(0)}=\operatorname{softplus}_T(-\boldsymbol D^{-1}\boldsymbol g^{\mathrm{step}})$，或把式 (26)/(28) 的 $T$ 标为 $T/h$。
- **Lemma 2：**保持已改的 "initial diagonal estimate"，不再声称最终冲量只依赖对角近似。
- **Assumption 2：**加一句 "The decoupling assumption is used only to construct the initial estimate; the correction in (29) reintroduces the coupling through $\boldsymbol W_{\mathrm{env}}$."
- **式 (34) 后：**"The impulse derivative follows from (29). For fixed contact geometry, $\boldsymbol D$, $\boldsymbol W_{\mathrm{env}}$ and $\hat L_N$ are constant, so the map is smooth in $\boldsymbol\lambda_{\mathrm r}$." 梯度链式展开为
  $$\frac{\partial\hat{\boldsymbol\beta}}{\partial\boldsymbol\lambda_{\mathrm r}}=\operatorname{diag}\big(\sigma'(\cdot)\big)\Big[(\boldsymbol I-\omega\boldsymbol D^{-1}\boldsymbol W)\frac{\partial\boldsymbol\beta^{(0)}}{\partial\boldsymbol\lambda_{\mathrm r}}-\omega\boldsymbol D^{-1}\frac{\partial\boldsymbol g^{\mathrm{step}}}{\partial\boldsymbol\lambda_{\mathrm r}}\Big].$$
- **式 (30)/(31)：**不改。式 (31) 中已划掉的 $\max(-\boldsymbol D^{-1}\boldsymbol s_i,0)$ 约束保持删除。
- **多步预测（$T>1$）时：**$\boldsymbol d_{\mathrm{env}}=\max(\tilde{\boldsymbol\phi},0)$ 也换成 $\operatorname{softplus}$，否则光滑性在步间断开。单步预测不受影响。

---

## 4. 可选的推广：$k$ 步展开

$$
\boldsymbol\beta^{(j+1)}=\operatorname{softplus}_T\!\Big(\boldsymbol\beta^{(j)}-\omega\boldsymbol D^{-1}\big(\boldsymbol W_{\mathrm{env}}\boldsymbol\beta^{(j)}+\boldsymbol g^{\mathrm{step}}_{\mathrm{env}}\big)\Big),\quad j=0,\dots,k-1.
$$

- $k=0$：只取对角逆 Delassus 作阻抗，对应 ComFree 式 (10) 的理想阻抗按对角取值。
- $k=1$：本文默认模型。
- $k\to\infty$：光滑化的阻尼 Jacobi 迭代，收敛到带 softplus 的 LCP 解。

每步多一次 matvec，仍闭式、光滑。论文默认 $k=1$，$k$ 只作为消融轴出现。

---

## 5. 必须重做的实验

- **Table IV：**用新式 (29) 重算四种接触状态的速度误差。
- **Table V：**重算 SCM 的 ideal/real 成功率与规划时间。
- **新增消融一行或一列：**$k=0$、$k=1$、旧 $\alpha$-clip 三组对比；若精度不够，再加软激活集 Newton 变体。
- **Fig. 4：**若图中画有投影或缩放框，改为 "Preconditioned Residual Correction"。

**不要沿用旧 $\alpha$-clip 模型的任何数值作为新模型结果。**

---

## 6. 不采用的方案及原因

| 方案 | 原因 |
|---|---|
| 把 clip 换成光滑饱和的 $\alpha$ 线搜索 | 一个标量共享给所有接触，只能缩放幅值，改不了接触间分配 |
| 非预条件 $1/\lambda_{\max}(\boldsymbol W)$（当前写法） | 需要特征值；最硬的行决定步长，其余行几乎不动 |
| 按接触解 3×3 或 4 面小 LCP | 离散激活集，非光滑 |
| Rayleigh 精确线搜索步 | 闭式但投影后不保证下降，留作备选 |
