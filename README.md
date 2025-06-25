# KnowFlowAI: 从入门到生产级的RAG系统演进

欢迎来到 KnowFlowAI 项目！本仓库通过三个逐步进化的Python脚本（`main1.py`, `main2.py`, `main3.py`），完整地展示了如何从零开始，一步步构建一个功能完备、可用于生产环境的检索增强生成（RAG）问答系统。


## 项目演进之路

本仓库的核心是三个主脚本，每一个都代表了RAG系统开发的一个关键阶段：

---

### 阶段一: `main1.py` - 基础RAG系统

这是梦开始的地方。`main1.py` 实现了一个最基础的RAG系统，帮助您快速理解核心概念。

- **功能**:
  -   知识库内置于代码中（硬编码）。
  -   对**整个文档**进行向量化和索引。
  -   实现了"检索-生成"的基本两步流程。
- **目标**: 学习RAG的基本工作原理，搭建一个最小化的原型。
- **详情**: 查看 [`readme1.md`](./readme1.md) 获取该脚本的详细说明。

---

### 阶段二: `main2.py` - 引入文档分块的进阶RAG

在基础版之上，`main2.py` 解决了一个核心痛点：当知识库文档很长时，直接对整个文档进行检索效率低下且不精确。

- **核心改进**:
  -   **文档分块 (Chunking)**: 在索引前将长文档切分成带有重叠部分的、更小的文本块。
  -   **精确检索**: 索引和检索的对象是**文本块**，这使得系统能更精确地定位到与问题最相关的具体信息。
  -   **优化上下文**: 将检索到的原始文档（提供宏观背景）和最相关的文本块（提供精确细节）共同构建成更丰富的上下文。
- **目标**: 提升检索的精确度和最终答案的质量。
- **详情**: 查看 [`readme2.md`](./readme2.md) 获取该脚本的详细说明。

---

### 阶段三: `main3.py` - 生产级RAG系统

这是本项目的最终形态，一个功能完备、代码健壮、可用于实际场景的RAG系统。

- **核心特性**:
  -   **动态知识库**: 无需修改代码，直接从本地`./docs`目录加载多种格式的文档（`.txt`, `.pdf`, `.docx`等）作为知识库。
  -   **答案可溯源**: 生成的答案所依赖的上下文会明确标注其**文件来源**，极大地增强了系统的可信度和实用性。
  -   **完全模块化**: 所有逻辑被封装在类中，代码结构清晰，易于维护和扩展。
  -   **智能缓存**: 自动缓存所有中间产物（模型、索引、映射关系），二次启动时可实现秒级加载。
- **目标**: 构建一个可随时部署、知识库可动态扩展的生产级RAG应用。
- **详情**: 查看 [`readme3.md`](./readme3.md) 获取该脚本的详细说明。

## 快速开始

### 1. 安装依赖

本项目依赖多个Python库。为确保所有脚本都能正常运行，请一次性安装所有必需的依赖：

```bash
# 核心依赖
pip install sentence-transformers faiss-cpu numpy openai

# 用于main3.py文件读取的依赖
pip install PyPDF2 python-docx pandas openpyxl markdown beautifulsoup4
```
**提示**: `faiss-cpu`是CPU版本。如果您的机器有NVIDIA GPU并配置了CUDA，可以安装`faiss-gpu`以获得更好的性能。

### 2. 配置API密钥

所有三个脚本都需要使用智谱AI（ZhipuAI）的API服务。请在运行脚本前，设置一个名为 `ZHIPUAI_API_KEY` 的环境变量。

**在 Linux 或 macOS 上:**
```bash
export ZHIPUAI_API_KEY="你的智谱AI_API_KEY"
```

**在 Windows 上:**
```powershell
$env:ZHIPUAI_API_KEY="你的智谱AI_API_KEY"
```

### 3. 运行脚本

您可以根据学习需要，选择运行任意一个脚本：

- **运行基础版:**
  ```bash
  python main1.py
  ```

- **运行分块版:**
  ```bash
  python main2.py
  ```

- **运行生产版:**
  - 首先，在项目根目录下创建一个`docs`文件夹，并放入你的知识库文件。
    ```bash
    mkdir docs
    # (然后将你的 .txt, .pdf, .docx 等文件复制到 docs 文件夹中)
    ```
  - 然后运行脚本:
    ```bash
    python main3.py
    ```

## 核心技术栈

- **文本嵌入**: `sentence-transformers` (使用 `moka-ai/m3e-base` 模型)
- **向量存储与检索**: `faiss` (Facebook AI Similarity Search)
- **大语言模型 (LLM)**: `openai` 库 (调用智谱AI的 `glm-4-plus` 模型)
- **文件处理**: `PyPDF2`, `python-docx`, `pandas` 等

