# 作者 SMORE+DAMPS 接入核查（2026-10-31）

结论：可访问的作者材料没有提供 SMORE 模型实现，因此无法确认论文中 SMORE+DAMPS 的实际接入路径。当前投影接口实验继续作为已明确声明的实现，补种子时保持算法与参数不变。

## 查验范围与证据

- 作者仓库 [Wmhwxl/KDD2026_DAMPS](https://github.com/Wmhwxl/KDD2026_DAMPS)，main HEAD `13dc1b8d1a9ee53fe0eca12ab2e4c8669f291cd0`。遍历全部 35 个可达历史提交的完整文件树，唯一含 SMORE 的路径是 `src/configs/model/SMORE.yaml` 和 `src/configs/models/SMORE.yaml`，没有 SMORE Python 文件。机器可读证据：`author_history.json`。
- 作者仅有 main 分支及 v1.0.0 tag；公开 release 无附加附件，公开 issues 和 PR 列表为空。README 提到 `-m SMORE`，但示例与 DAMPS 开关说明针对 MGCN；README 提及模型名称不等于提供了实现。
- [Zenodo 18028194](https://doi.org/10.5281/zenodo.18028194) 当前解析至版本 18028195，归档 `Wmhwxl/KDD2026_DAMPS-v1.0.0.zip`，324283 bytes，MD5 `984cedc5e0354f91d59db23cce9fabe2`。归档内也只有 SMORE.yaml。DAMPS 文件 SHA256 为 `f0183b7bf9fa238f3ffea1af6a86f5fba6d193f6c81a961fc9338e8a1062a9a9`，与本项目和同组副本一致。
- 用户确认 [ustc-mkh/KDD2026_DAMPS](https://github.com/ustc-mkh/KDD2026_DAMPS) 是组内修复后的作者代码，当前 main 仍为 `cafcc930546799f5ca393ad515fbd5c7b6d85e6d`，模型目录未提供 SMORE。服务器上的旧 COMP5331 副本只有 MGCN；另行核查在线 [COMP5331](https://github.com/ustc-mkh/COMP5331) main `11788271f2a01067ab175771384c06510f8a1554`，已有 LIRDRec，但仍没有 SMORE。
- 在线 COMP5331 的 `src/mmrecsys/nn/damps.py` Git blob 为 `fe5c0755ccdc16fbbf01ca32cfbc5183ea3c362d`，与本项目 `src/common/damps_task1.py` 完全相同。共享 DAMPS 提供纯投影接口和内部独立投影两种路径：MGCN 使用后者，当前 LIRDRec 在其原有模态投影后使用前者。这支持组内接口兼容性，不能替代作者 SMORE 接入证据。
- SMORE 官方实现与 MMRec 的 SMORE 文件相同，属于 raw backbone；它们不能证明 DAMPS 作者的接入方案。

## 确认到的路径差异

| 项目 | 作者公开 MGCN+DAMPS | 本轮 SMORE+DAMPS |
|---|---|---|
| 模态表示来源 | DAMPS 内部 embedding 与独立 Linear 投影 | SMORE 已有 embedding 与 Linear 投影 |
| forward 传入的特征 | 数值被忽略，仅用形状确定 IFFT 长度 | 实际参与 FFT 和校准 |
| 下游位置 | MGCN gate_v/gate_t 之前 | SMORE 原有 spectrum_convolution 之前 |
| 外部投影的推荐损失梯度 | 该路径不回传到外部投影 | 回传到 SMORE 投影 |
| 证据性质 | 有 MGCN 源码；不能直接外推至 SMORE | 本项目已执行且测试过的显式设计 |

作者 MGCN 当前 HEAD 还在构造函数中引用未定义的 image_feats/text_feats；团队 fork 已修复相关初始化。核查使用源码路径和算子测试，不将公开 MGCN 当前文件的可运行性或路径当作作者 SMORE 证据。

公开 DAMPS 保留的 AVRF logit 权重、APC 旋转符号和 IMCF 逐元素比值，与上一轮记录一致。尚无证据说明更换接入路径能解释或修复无提升结果；不将推测的新路径混入同一组三种子的比较。

## 固定的后续验证

使用 seeds 2024、2025，对五个数据集分别运行 Raw/Full，共 20 次；与现有 seed 999 合成三种子配对对照。保持模型、数据、full-ranking 评价、优化器、学习率、early stopping、backbone 参数及 DAMPS 接入方式不变。`fixed_protocol.json` 锁定模型/训练源码、配置和第一轮数据指纹；`plan_seeds_20261031.json` 固定任务清单。当前不额外扩展三个消融的种子，也不启用猜测的作者路径。

比较同时报告每个 seed、均值、样本标准差和配对差值；不得用最好的 seed 代替三种子结果。结果确认后才提出 PR。
