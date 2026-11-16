# 区间计数条件分析

本文件公开新增功能的完整契约；现有正常归档、普通分析、结果读取和错误边界见 README.md。新增能力尚未实现。保留全部原接口、历史普通结果、原测试及 baseline-lock.json 锁定的参照；可以增加生产实现和新测试，不得改写原测试或冻结参照。新功能不改变普通 analyze 的序列长度能力。

## 共同业务结果与两项交付

新增计数条件推断及其可消费表示；窗口分析消费同一事件后验计算占用矩，并投影顺序证据。两者共同保证证据与报告均以完整观测和全部计数条件为条件。

令 X_t 为时刻 t 的隐状态，O 为整个归档观测，N 为其长度。每条计数条件约束某个左闭右开区间内指定状态恰好出现若干次。事件 E 是所有条件的交集，而不是逐时刻筛选。相邻时刻原状态边际并不足以描述一般事件后验，不能投影后重新假设原状态仍构成一阶条件链。

两项交付须联合核验事件质量、逐时刻证据、各窗均值及完整协方差，并说明推断到窗口分析交接所保持的不变量。允许全量重算、逐窗重求和复用现有数值设施，不要求长序列资源优化。具体内部表示由实现选择；不要求两个独立服务、并发工作者或跨进程任务续接。

## Python 调用与输入

从 analysis_app 导出 CountCondition、CountAnalysis，继续导出原有名称。

CountCondition(start, stop, state, count) 为冻结数据类，四个同名公共字段按该位置顺序构造；构造只保存值，资格在 analyze_counts 中统一核验。

新增调用：
    app.analyze_counts(recording_id, model_id, conditions, windows, evidence_sink)

沿用 App(path) 及上下文管理；返回和普通 analyze 共用 analysis-N 编号空间的分析 ID 字符串。归档和模型仍使用已有登记入口。conditions 为可迭代对象，本次调用须取得独立快照，含 1 或 2 个 CountCondition；不接受以普通元组代替单项对象。保存条件的输入顺序和重复项。

仅新增入口限制 0≤N≤64；模型沿用 2–8 状态及现有概率登记规则。每项 start、stop、count 均须为非布尔 int，满足 0≤start≤stop≤N 和 0≤count≤min(8,stop-start)；state 须为模型中已登记的状态字符串。未知条件状态是 ValueError。空区间只能计数 0。

条件可重叠、重复或互相矛盾；一个时刻须同时计入所有适用条件。重复条件不重复乘事件概率；相矛盾的合法条件是零联合质量，不是输入格式错误。零次条件禁止相应区间内该状态出现。条件之外的时刻也受整个 O 和 E 共同约束。

windows 完全沿用 Window、至多16窗、名字唯一、两个不同状态及状态输出顺序等原规则。允许无窗口、空窗口及条件区间之外的报告窗口。evidence_sink 沿用可调用对象要求，按时间升序每个时刻调用一次 Evidence；概率仍按原模型状态顺序，不暴露内部记忆标签。

## 返回、数值及持久化

get_analysis 对新增入口发布的 ID 返回冻结 CountAnalysis；它保留 Analysis 的全部公共字段，另有 conditions 和 log_event_probability。conditions 是由冻结 CountCondition 组成的元组。states、reports 以及报告内部容器继续为元组，不持有输入可变别名。关闭再打开数据库仍返回同样类型和值；普通分析仍返回原 Analysis。

log_likelihood 表示 log P(O,E)，log_event_probability 表示 log P(E|O)，均为自然对数。两者须使用模型登记后的概率。结构零保持零；正联合质量即使小到浮点概率下溢，也不能按零事件拒绝，须返回有限的对数值。

逐时刻证据为 P(X_t=s|O,E)。每窗的第 a 个占用量 C_a 是该窗指定状态出现的次数，mean[a]=E[C_a|O,E]，covariance[a][b]=E[C_a*C_b|O,E]-mean[a]*mean[b]；包含跨时刻相关项，空窗全为零。本题不要求不同报告窗口之间的联合协方差矩阵。所有数值结果逐项绝对和相对容差均为 2e-10，以 math.isclose 的任一容差满足为准；不得仅比较总和或格式化字符串。

CountAnalysis 保存本次输入绑定与规范条件快照。输入容器之后被修改、再次分析和关闭重开都不得改变已发布结果。旧普通报告及新计数报告可以共存，失败重试不消耗分析 ID。

## 异常和交付边界

非法输入抛 ValueError，未知归档或模型抛 KeyError，零联合质量抛 ImpossibleSequence；以上均不得交付证据或发布。所有条件、窗口、序列长度、字母表及接收端资格在交付前核验；同时有多种错误时不约定优先级。

证据接收端异常原样传播，已交付前缀不可撤回，此次不发布；全部证据交付后才事务发布完整结果。数据库发布失败原样传播并回滚，旧结果可读，新结果不可见，重试不消耗失败分析的 ID。沿用 README.md 的串行调用及回调不重入规则，不新增并发恢复要求。

## 公开样例与核验入口

count_examples.py 提供普通输入夹具和手算期望，不实现计数推断。load_case(app, name) 登记模型、归档观测，返回 (conditions_specs, windows)；conditions_specs 每项是用于 CountCondition(*spec) 的参数元组。EXPECTED 中的有理数均为 [分子,正分母]，marginals 的列顺序为模型状态顺序；mean、covariance 遵循各 Window.states 顺序。log_event_probability 是 event_probability 的自然对数；rare 的该值直接以 log 表达，避免样例自身下溢。

once：三个独立公平 off/on 时刻全部缺测，全段 on 恰好一次。P(E|O)=3/8，各时刻 on 概率1/3；全段按(on,off)顺序，均值(1,2)，协方差全零。

overlap：四个独立公平时刻全部缺测，条件为[0,3)的on恰好一次、[1,4)的on恰好一次。P(E|O)=3/16，各时刻on概率1/3；全段均值(4/3,8/3)，协方差((2/9,-2/9),(-2/9,2/9))。

conflict：同一全段on计数分别要求0和1，须抛ImpossibleSequence且不交付。empty：空序列配[0,0)的on计数0，两个对数均为0，无证据、空窗矩为零。rare：两个缺测时刻，off概率均为1e-250，要求off出现两次；联合质量为正，对数约为2*log(1e-250)，条件证据全为off，全段(on,off)均值(0,2)，协方差全零。

python3.11 -B count_examples.py --check 仅核验夹具及原普通分析控制值，不能代替新增能力验收。python3.11 -B checks.py 是现有完整正常核验；python3.11 -B demo.py 演示原归档和重开流程。

补独立短路径枚举对照，逐项核对联合质量、事件概率、证据及各窗完整矩，覆盖重叠、重复、冲突、结构零、末尾硬证据、极小正质量、空序列与空窗；联合核验接收端异常、数据库发布失败、旧结果保全及真实文件重开。覆盖 N=64、8状态、两条件计数均达8及重叠报告的合法输入，不要求枚举该规模的路径，也不添加资源达标门槛。

沿现有 .pointwise-runtime.json 保留 version=1、kind=cli 和 setup 的 argv 数组列表格式；verify、demo 各为可执行 argv 数组。verify 保留全部原检查并接通新增独立核账及发布边界核验，demo 展示计数事件如何同时改变证据和窗口不确定性，并关闭重开读取结果。入口名称由交付者选择，不能仅打印成功。无需运行 bench.py 的正式长负载作为本题新增验收。
