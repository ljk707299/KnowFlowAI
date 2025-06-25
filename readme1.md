# RAG (Retrieval-Augmented Generation) 系统

这是一个基于检索增强生成（RAG）技术的简单问答系统实现。该项目使用 `sentence-transformers` 模型将文本知识库向量化，并通过 `faiss` 进行索引，以实现高效的相似性检索。当用户提出问题时，系统会首先从知识库中检索最相关的文档，然后将这些文档作为上下文，利用大型语言模型（LLM，本项目使用智谱AI的GLM-4）生成精准的回答。

## 功能特性

- **本地知识库**: 支持使用自定义的文本列表作为知识库。
- **高效检索**: 使用 `faiss` 库进行快速、高效的向量相似度搜索。
- **模型本地化**: 支持从 Hugging Face 下载 `sentence-transformers` 模型并缓存到本地，提高后续运行速度。
- **安全配置**: 通过环境变量加载 API 密钥，避免敏感信息硬编码。
- **模块化设计**: 代码被封装在 `RAGSystem` 类中，结构清晰，易于扩展。

## 项目结构

```
.
├── main1.py            # 主程序脚本
├── readme1.md          # 项目说明文件
├── local_m3e_model/    # (自动生成) 本地缓存的嵌入模型
└── m3e_faiss_index.bin # (自动生成) Faiss 索引文件
```

## 安装依赖

在运行此项目之前，您需要安装所有必需的 Python 库。您可以通过 pip 来安装它们：

```bash
pip install sentence-transformers faiss-cpu numpy openai
```
**注意**:
- `faiss-cpu` 是 CPU 版本的 Faiss。如果您的机器支持并配置了 CUDA，可以安装 `faiss-gpu` 以获得更好的性能。
- `openai` 库用于与智谱AI的API进行交互。

## 配置

### API 密钥

本项目需要使用智谱AI（ZhipuAI）的 API 服务。请在运行脚本前，设置一个名为 `ZHIPUAI_API_KEY` 的环境变量，并将其值设置为您的有效 API 密钥。

**在 Linux 或 macOS 上:**
```bash
export ZHIPUAI_API_KEY="你的智谱AI_API_KEY"
```

**在 Windows 上:**
```powershell
$env:ZHIPUAI_API_KEY="你的智谱AI_API_KEY"
```

## 如何运行

配置好环境变量后，直接在终端运行主脚本即可：

```bash
python main1.py
```

程序首次运行时，会自动从 Hugging Face 下载 `moka-ai/m3e-base` 模型并创建 Faiss 索引文件。后续运行将会直接加载本地缓存的模型和索引，启动速度会更快。

## 代码逻辑解析

`main1.py` 的核心逻辑分为两个部分：**初始化** 和 **问答流程**。

### 1. 初始化 (`RAGSystem.__init__`)
当 `RAGSystem` 被实例化时，会执行以下操作：
1.  **加载嵌入模型**: `_load_model` 方法会检查本地是否存在模型文件。如果不存在，则从网络下载并保存。
2.  **创建/加载索引**: `_create_or_load_index` 方法会检查本地是否存在 Faiss 索引文件。如果不存在，它会：
    -   使用加载的模型为知识库中的所有文档生成向量嵌入。
    -   创建一个新的 Faiss 索引。
    -   将文档嵌入添加到索引中。
    -   将索引保存到本地文件。
3.  **初始化LLM客户端**: 使用环境变量中的 API 密钥初始化与大模型服务的连接。

### 2. 问答流程 (`main` 函数)
1.  **定义知识库**: 在 `main` 函数中，我们定义了一个包含几段文本的 `documents` 列表作为我们的本地知识库。
2.  **定义查询**: 设置一个我们想要提问的问题。
3.  **检索 (Retrieval)**:
    -   `retrieve_docs` 方法首先将用户查询转换为向量。
    -   然后使用该查询向量在 Faiss 索引中搜索最相似的 `k` 个文档向量。
    -   返回这些向量对应的原始文档。
4.  **生成 (Generation)**:
    -   `generate_answer` 方法将上一步检索到的文档组合成一个"上下文（Context）"。
    -   构建一个包含上下文和原始问题的提示（Prompt）。
    -   将这个提示发送给大型语言模型（GLM-4），并获取生成的答案。
5.  **输出结果**: 打印检索到的文档和最终生成的答案。

这个流程清晰地展示了 RAG 如何结合检索和生成技术，为用户提供基于特定知识库的、更可靠的回答。 