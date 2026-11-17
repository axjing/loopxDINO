# AGENTS.md

## 🤖 AI 编程助手指南 (AI Agent Coding Guidelines for loopxDINO)

你好，AI 智能体。这是为了帮助你更好地理解和开发 `loopxDINO` 项目而准备的文档。在编写或修改任何代码之前，请仔细阅读这些约束条件。

## 📌 核心开发理念与硬件约束

*   **硬件画像**: 目标运行环境是单张具有严格 **16GB VRAM** 显存和 32GB 系统内存的 NVIDIA RTX 5060 Ti 显卡，运行在 Windows 11 原生环境（或 WSL2 Ubuntu）下。
*   **核心理念**: **计算深度优于参数广度 (Compute depth over parameter width)。** 我们使用基于循环操作的隐式潜在空间推理 (System 2)，而不是无脑堆叠深层 Transformer 层。
*   **模块化原则**: 严格解耦。感知底座 (System 1) 绝对不能与推理解码器 (System 2) 纠缠在一起。

## 🛠️ 环境与命令设定 (Setup commands)

*   **安装依赖**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` 然后 `pip install transformers accelerate einops`
*   **运行测试**: 优先运行项目根目录下的测试脚本，例如：`python scripts/test_decoder_vram.py`。
*   **代码格式化**: 遵循 PEP 8 标准，推荐使用 `black` 或 `ruff` 格式化代码。

## 🚨 严格编码规范 (Strict Coding Rules - MUST FOLLOW)

### 1. 显存管理 (Memory Management - VRAM is King)
*   **DINOv3 是只读的 (READ-ONLY)**: DINOv3 底座必须**始终**以 `eval()` 模式加载。它的所有参数必须严格设置 `requires_grad = False`。永远不要为底座计算梯度。
*   **强制使用检查点 (Checkpointing is Mandatory)**: 在 `LoopedCrossAttentionDecoder` 内部，`for step in range(num_loops):` 循环会在时间反向传播 (BPTT) 期间创建一个很深的计算图。你**必须**使用 `torch.utils.checkpoint` 包裹循环内的 `MultiheadAttention` 和 `FFN` 模块调用，以保持激活值显存占用为 $O(1)$。
    *   *切勿在没有检查点保护的情况下展开循环。*
*   **精度要求**: 默认使用 `torch.bfloat16`。如果调用硬件特性，请使用 `torch.float8_e4m3fn` (FP8)。

### 2. 算法完整性 (Algorithmic Integrity)
*   **循环步嵌入 (Loop-step Embedding)**: 不要将完全相同的 Query 在未修改的情况下传入连续的循环中。你**必须**在每次循环迭代开始时，向 Query 中注入一个 `step_embedding` (例如，`step_embed(torch.tensor(step))`)。这能防止表示坍塌 (representation collapse) 或过度平滑 (over-smoothing)。
*   **KV 缓存复用 (KV Caching)**: 从 DINOv3 派生出的 Key 和 Value 在各个循环中是静态的。在循环前只计算**一次**，并将它们作为 `memory` 传递给 `CrossAttention`。不要在推理循环内部重新计算视觉特征。

### 3. 代码风格与标准 (Style & Standards)
*   **语言**: Python 3.12+。
*   **偏好原生实现**: 优先使用 PyTorch 原生实现，避免使用沉重的第三方抽象库。直接使用带有 `batch_first=True` 的原生 `nn.MultiheadAttention`，或者在支持的情况下直接使用 `flash-attn`。
*   **张量设备指派**: 确保所有的张量直接在正确的 `device` 和 `dtype` 上创建，避免在实例化张量后进行昂贵的 `.to(device)` 调用。

## 🏗️ 架构蓝图 (Architecture Blueprint)

*   `core/backbone.py`: 封装和冻结 `facebook/dinov3-*` 模型的逻辑。
*   `core/decoder.py`: 单层循环交叉注意力解码器 (`LoopedCrossAttentionDecoder`)。
*   `core/model.py`: 整合底层底座和解码器的包装类 `LoopxDINO`。
*   `train.py`: 极简训练循环，处理梯度累加和 FP16/BF16 混合精度缩放。

## 📄 开源许可证规范 (License & Open Source Compliance)

*   **项目协议**: 本项目遵循 **Apache License 2.0** 开源协议，完整条款详见根目录 [LICENSE](LICENSE) 文件。
*   **合规要求**: 编写及引入代码、依赖或模型组件时，必须遵守 Apache 2.0 协议规定，严禁引入与 Apache 2.0 协议冲突的受限专有代码或未经授权的闭源资产。

**在生成代码前请务必确认**：反复检查你提议的修改是否会违反 16GB VRAM 的约束。如果某项操作导致显存占用与 `num_loops` 呈线性增长关系，请立即重写该操作。