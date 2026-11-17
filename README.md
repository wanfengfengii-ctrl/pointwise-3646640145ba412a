# Sequence Archive

Python 3.11 标准库实现的有限状态观测分析组件。它将有序观测归档、不可变模型登记、完整序列后验、区间占用量和发布结果串成同一流程。SQLite 文件由调用者指定；不需要安装依赖。

运行：

```sh
python3.11 -B checks.py
python3.11 -B demo.py
python3.11 -B count_demo.py
python3.11 -B bench.py baseline
python3.11 -B bench.py compare
```

checks.py 运行全部原回归、参照完整性检查、独立计数条件短路径核账和两个小规模的实际端到端计量。demo 在测试临时目录创建真实数据库，归档设备观测，打印顺序证据及报告，关闭后重新打开读取发布结果。count_demo 在同一归档上对比普通后验与“on 恰好出现一次”的计数条件后验，展示证据和窗口协方差如何同时被事件改变，再关闭重开读取冻结 CountAnalysis。bench 的正式两档是每类 10000、40000 时刻，各三轮；它输出实际耗时和 Python 分配峰值，不把正常物化实现解释成低内存实现。所有临时目录均由标准库生成并自动清理。

## 区间计数条件

`CountCondition(start, stop, state, count)` 与 `App.analyze_counts(recording_id, model_id, conditions, windows, evidence_sink)` 提供以一至两个左闭右开占用计数条件交集为事件的推断；新入口限 64 时刻，发布冻结 `CountAnalysis`（含 Analysis 全部字段、`conditions` 与 `log_event_probability`），`log_likelihood` 为 log P(O,E)。完整契约、手算样例和精度见 COUNTS.md，内部设计与交接见 COUNTS_HANDOFF.md，独立枚举核账为 count_audit.py，夹具样例为 count_examples.py。普通 analyze 接口、历史结果和原测试保持不变。

## 公开接口

```python
from analysis_app import App, Window

with App("observations.sqlite") as app:
    app.register_model("mode", ("idle", "active"), ("low", "high"),
                       (0.5, 0.5), ((0.9, 0.1), (0.2, 0.8)),
                       ((0.95, 0.05), (0.1, 0.9)))
    app.archive_recording("shift", ("low", None, "high"))
    key = app.analyze("shift", "mode",
                      [Window("early", 0, 2, ("idle", "active"))],
                      lambda evidence: print(evidence.time, evidence.probabilities))
    result = app.get_analysis(key)
```

App 支持上下文管理和 close；仅支持单工作者串行调用。回调不得重入同一个 App 或修改其底层连接。调用方交给归档和模型登记的可变输入会被复制。结果为冻结数据类及元组，不持有数据库游标。

- archive_recording(recording_id, observations)：接受有限有序的非空字符串或 None 序列，允许空序列。None 占据一个时间位置。不接受字符串作为整个序列。ID 和符号不得为空或含 NUL。相同 ID、相同内容幂等；冲突拒绝，无部分归档。归档尚未绑定模型，字母表资格在分析时核验。
- register_model(model_id, states, symbols, initial, transitions, emissions)：登记 2–8 个互异状态、至少一个互异观测符号。维度依次为 K、K×K、K×L。每个概率接受 int/float，不接受 bool；必须有限非负。每行总和与 1 的绝对误差不超过 1e-12，接纳时除以行和归一化，结构零不变。模型 ID 一经登记不能替换；重复登记也拒绝。
- analyze(recording_id, model_id, windows, evidence_sink)：至多 16 个具名左闭右开窗口；名字互异，0 ≤ start ≤ stop ≤ N。每窗指定两个不同的已登记状态，状态顺序决定输出两维顺序。返回 analysis-N 形式的分析 ID。每个时刻恰好调用一次接收端，参数 Evidence(time, probabilities)，按时间升序；概率顺序为模型状态顺序。返回后可通过 get_analysis 读取绑定、长度、对数似然、每窗均值及完整 2×2 协方差。无窗口仍交付全部证据；空窗口的矩为零。
- get_analysis(analysis_id)：读取已完成发布的 Analysis。归档、模型和发布报告跨数据库关闭与重开保留。再次分析产生新 ID，旧报告及其绑定不变。

初始分布作用于时刻 0，随后先转移再发射。缺测的发射因子为 1。每条证据及每份窗口报告均以整个观测序列为条件，窗口外的后续硬证据可以改变早期窗口。占用量是该窗口内各指定状态出现的次数；协方差包含不同时间之间的相关项。两个等概率且永久保持原状态的状态、两个缺测时刻，其占用次数均值为 (1,1)，协方差为 ((1,-1),(-1,1))。

## 错误与发布边界

非法输入抛 ValueError，未知归档/模型/分析 ID 抛 KeyError。序列似然为零抛 ImpossibleSequence，证据接收端不会收到任何行，也不会发布报告。所有窗口与字母表在证据交付前验证。

接收端异常原样传播，可能已经交付的前缀不可撤回，此次分析不发布；调用者重试必须使用能处理新一轮完整证据的接收端。接收端不是事务参与者。证据全部交付后才以 SQLite 事务提交完整报告；提交失败原样传播，旧报告可读，新报告不可见，重试不消耗失败分析的 ID。实际数据库触发器中止、回滚、重开及重试见 tests/test_failures.py 和 tests/test_publication.py。非法归档或模型不写入部分对象，源迭代异常也原样传播。应用不捕获并伪装数据库或文件系统错误。

## 实现与数值回归

recordings.py 和 models.py 持有归档状态；inference.py 使用归一化对数域前后向计算，完整物化每时刻状态后验及相邻两时刻联合后验。occupancy.py 每份报告复制物化表，并按后验条件链传播一阶和二阶原始矩；它没有将逐点方差相加。publication.py 将同一次推断交给报告和接收端，最后发布完整绑定。

reference.ReferenceApp 直接调用保留的 materialize、materialized_reports；正常生产 infer、summarize 当前委托同一实现。baseline-lock.json 冻结参照函数的 AST 摘要。独立正确性参照在 tests/oracle.py：以 Fraction 枚举短序列全部状态路径，不调用生产推断或矩计算。原回归还覆盖结构零、末尾硬证据、极小似然、空序列、重叠窗、状态次序、输入快照和最大状态数。短例逐项绝对/相对容差各 2e-10，长负载对参照逐项绝对/相对容差各 2e-9；不以总和或四舍五入字符串替代逐项判定。

## 资源基线

bench.py 的 dense 类含四状态、非零转移及周期缺测；sparse 类含五状态、结构零、缺测及 1e-250 量级发射。三份窗口包含全长和重叠区间。生成、入库、打开夹具和数据库不计入；计量从 analyze 读取归档开始，到 get_analysis 和接收端完整核验结束，包含推断、报告、证据交付和发布。接收端顺序读参照二进制夹具并丢弃，不替生产实现保留证据。参照计量写同样的二进制行；两侧都是实际文件 I/O。

baseline 模式执行冻结正常实现。compare 模式额外执行生产入口，逐行核验证据、逐项核验窗口报告，再输出两侧三轮中位值。完整负载较慢；--smoke 仅用于夹具自检，不代表正式资源结果。正常参照不创建自有暂存文件，参照暂存量为 0，夹具、已归档输入数据库和已发布结果不属于自有暂存。compare 对未来任意实现的精确暂存峰值输出 null，不能据此宣称资源准入。若改造使用暂存，须由独立平台计量全部自有暂存及其清理；本计量器不会用目录轮询下界冒充真实峰值，也不提供未经测量的资源达标断言。
