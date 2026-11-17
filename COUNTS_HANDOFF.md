# 计数条件分析交接说明

本说明覆盖 `App.analyze_counts` 与 `CountAnalysis` 的内部设计、不变量、持久化和
核验入口。业务契约以 COUNTS.md 为准；普通分析路径、原测试和 baseline-lock.json
锁定的参照函数均未改动。

## 组件划分

- `analysis_app/counts.py`
  - `CountCondition`、`CountAnalysis`：冻结数据类，字段和构造顺序见 COUNTS.md。
  - `validate_conditions`：对可迭代条件取独立元组快照并统一资格核验；保存输入
    顺序与重复项，非布尔 int、已登记状态、区间和 `0..min(8, 区间长度)` 均在此
    拦截。窗口仍复用 `occupancy.validate_windows`。
  - `constrain`：增广链上的对数域前向后向，产出 `_CountedPosterior`。
  - `counted_reports`：在事件后验收敛的增广链上逐窗传播一阶、二阶原始矩。
- `analysis_app/publication.py`
  - `analyze_counts`：先做全部输入与长度（N≤64）核验，再推断、交付逐时刻证据，
    最后在单个 SQLite 事务中发布；失败不消耗 analysis 编号。
  - `get_analysis`：按载荷 `kind` 分发，计数报告重开后仍是 `CountAnalysis`，
    普通报告仍是原 `Analysis`，两者共用 analysis-N 编号空间。
- `analysis_app/inference.py`、`occupancy.py`、`reference.py` 未修改，冻结 AST
  摘要继续通过 `bench.verify_frozen`。

## 增广链与事件不变量

令事件 E 为一至两个计数条件的交集。增广状态

    Y_t = (X_t, c_1(t), c_2(t))

其中每个计数器只在自己的左闭右开区间内、隐状态等于条件状态时加一，增量是
(t, X_t) 的确定函数。因此 Y 过程仍是一阶马尔可夫链；维度为
K·(count_1+1)·(count_2+1)，单条件时第二维为 1，最大 8·9·9=648 个增广状态。

前向 `alpha[t][u]` 与后向 `beta[t][u]` 全部保留自然对数值：

- 转移只走向不超过目标计数的后继；目标态只在末时刻被 beta 接受。
- `log P(O,E) = logsum_u(alpha[N-1][u] + beta[N-1][u])`（beta 末值对达标态为
  0），该值为 `-inf` 时抛 `ImpossibleSequence`，此前不交付任何证据。
- 普通 `materialize` 先跑一遍：既核验字母表和普通零似然，也提供
  `log P(O)`；`log P(E|O) = log P(O,E) - log P(O)`。
- 逐时刻证据由
  `P(X_t=s|O,E) ∝ logsum_{u: state(u)=s}(alpha+beta)` 归一化得到，状态列顺序
  仍是模型登记顺序，不暴露计数器标签。

正联合质量全程不转概率，只在最终归一化时做指数差；因此概率本身下溢（例如
count_audit 的 1e-162 双因子）时两个对数值仍有限，结构零仍由 `log(0)=-inf`
保留。重复条件是两个独立维度但目标相同，事件概率不会被重复相乘；互相矛盾的
合法条件自然收敛到零质量而不是输入错误。

## 窗口矩如何保留跨时间依赖

COUNTS.md 明确要求不能把事件后验投影回相邻原状态边际再假设一阶链。
`counted_reports` 不做这种投影：它直接在增广链 Y 上传播。

- 窗起点质量 `mass[u] = P(Y_t0=u | O, E)`，由 `alpha+beta-log P(O,E)` 得到。
- 窗内每步用事件条件核
  `P(Y_{t+1}=v | Y_t=u, O, E) ∝ T·emit·beta[t+1][v] / beta[t][u]`，按 u 归一。
- 联合占用一阶、二阶原始矩随 `q = P(Y_t=u,Y_{t+1}|O,E)` 递推；q 本身含完整
  前向后向信息，事件引入的任意跨时刻相关都保留在增广状态的计数器里。
- 均值 `E[C_a|O,E]` 与 `E[C_a C_b|O,E] - mean_a mean_b` 在窗末汇总；空窗为
  零。仅清理 roundoff 量级的残差（负对角夹零、近反对称的非对角取对称平均），
  不改变真实协方差。

## 持久化与边界

- 载荷是紧凑 JSON（`allow_nan=False`），含 `kind="counts"`、规范条件快照、
  两个对数值、状态与报告元组；重开时重建冻结 `CountCondition`/`Occupancy`。
- 输入容器在发布前已复制，发布后的外部修改、再次分析和关闭重开都不影响旧
  报告；普通 `Analysis` 与 `CountAnalysis` 可在同一数据库共存。
- 非法输入 ValueError、未知绑定 KeyError、零联合 ImpossibleSequence 均在证据
  交付前抛出；接收端异常原样传播且不发布；发布事务失败回滚、旧结果可读、
  重试不跳号。这些边界由 tests/test_counts.py 与 count_audit.py 联合覆盖。

## 核验与演示入口

- `python3.11 -B checks.py`：冻结参照校验、全部 unittest（含 14 个新计数用例）、
  独立 `count_audit.py` 短路径核账，以及原小规模资源 compare。
- `python3.11 -B count_audit.py`：Fraction 全路径枚举，逐项核对联合质量、事件
  概率、证据和完整窗口矩；覆盖重叠、重复、冲突、结构零、末尾硬证据、极小正
  质量、空序列/空窗，以及接收端异常、数据库中止、旧报告保全和真实文件重开。
- `python3.11 -B count_demo.py`：同一归档上对比普通与事件条件后验，展示证据
  与窗口协方差同时改变，随后关闭重开读取冻结 `CountAnalysis`。
- `python3.11 -B count_examples.py --check`：仅夹具与普通分析控制值，保持
  COUNTS.md 描述的定位。
- N=64、8 状态、两条件计数均为 8 的合法形状在 tests/test_counts.py 中覆盖；
  该规模不枚举路径，事件概率另用概率空间计数器独立核对。

## 已知限制

- 新入口按契约限制 N≤64 且条件 1～2 个；普通 `analyze` 的长度能力不变。
- 增广表全量物化（alpha/beta 各 N 张表），按 COUNTS.md 不做长序列资源优化，
  也没有跨进程续接或并发恢复。
- 不提供不同报告窗口之间的联合协方差（契约未要求）。
