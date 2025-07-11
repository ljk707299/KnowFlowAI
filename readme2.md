# 高级RAG系统：结合文档分块技术

本项目是 `main1.py` 的进阶版本，实现了一个更精细化的检索增强生成（RAG）问答系统。核心改进在于引入了**文档分块（Chunking）**机制。通过将大型文档切分成有意义的小块进行索引，系统能够更精确地定位与用户问题最相关的具体信息，从而为大型语言模型（LLM）提供更高质量的上下文，生成更准确的答案。

## 功能亮点

- **文档分块与重叠**: 在索引前自动将长文档切分为固定大小、有重叠的文本块，确保语义的连续性。
- **两级检索**: 检索时首先找到最相关的**文本块**，然后追溯其所属的**原始文档**。
- **优化上下文构建**: 将检索到的原始文档（提供宏观背景）和高相关的文本块（提供具体细节）共同构建为更丰富的上下文。
- **持久化缓存**: 不仅缓存模型和Faiss索引，还将文档与块的映射关系持久化存储，进一步加快后续启动速度。
- **完全封装**: 所有逻辑均封装在 `RAGSystemWithChunking` 类中，代码结构清晰，易于维护和扩展。
- **详细的执行日志**: 脚本运行时会打印详细的步骤信息，方便用户理解系统内部的每一个环节。

## 项目结构

```
.
├── main2.py                    # 主程序脚本
├── readme2.md                  # 本项目说明文件
├── local_m3e_model/            # (自动生成) 本地缓存的嵌入模型
├── m3e_faiss_chunk_index.bin   # (自动生成) 基于文本块的Faiss索引文件
└── chunks_mapping_data.npy     # (自动生成) 包含所有块和映射关系的数据文件
```

## 安装依赖

确保您已安装所有必需的Python库。如果尚未安装，请通过pip安装：

```bash
pip install sentence-transformers faiss-cpu numpy openai
```
**注意**:
- `faiss-cpu` 是CPU版本的Faiss。如果您的机器支持并配置了CUDA，可以安装`faiss-gpu`以获得更好的性能。
- `openai`库用于与智谱AI的API进行交互。

## 配置

### API 密钥

本项目需要使用智谱AI（ZhipuAI）的API服务。请在运行脚本前，设置一个名为 `ZHIPUAI_API_KEY` 的环境变量，并将其值设置为您的有效API密钥。

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
python main2.py
```

首次运行时，脚本会自动执行以下操作：下载模型、对文档进行分块、生成所有块的嵌入、构建Faiss索引，并将所有生成的文件（模型、索引、映射数据）保存到本地。后续运行将直接加载这些缓存文件。

## 代码逻辑深度解析

`main2.py`的核心逻辑比基础版更为复杂，主要体现在**索引构建**和**信息检索**两个阶段。

### 1. 初始化与索引构建 (`_create_or_load_index_and_mappings`)

这是系统的准备阶段，当找不到本地索引文件时，会执行以下一系列操作：

1.  **文档分块**:
    -   遍历知识库中的每一个文档。
    -   调用 `_chunk_document` 方法，将长文档按照预设的`CHUNK_MAX_CHARS`（最大字符数）和`CHUNK_OVERLAP`（重叠字符数）切分成多个文本块。重叠部分是为了防止信息在切分处被割裂，保证了上下文的完整性。

2.  **建立映射关系**:
    -   在分块的同时，系统会实时记录两种映射关系：
        -   `document_to_chunks`: 从**文档ID**到其包含的**所有块ID列表**的映射。
        -   `chunks_to_document`: 从**单个块ID**到其所属的**父文档ID**的映射。
    -   所有生成的文本块都存储在 `all_chunks` 列表中。

3.  **嵌入与索引**:
    -   使用加载的`m3e-base`模型，为 `all_chunks` 列表中的**每一个文本块**生成向量嵌入。
    -   将这些**块的嵌入向量**添加到Faiss索引中。这里的关键是，**索引的对象是块，而不是整个文档**。

4.  **持久化存储**:
    -   将构建好的Faiss索引保存为`.bin`文件。
    -   将包含`document_to_chunks`、`chunks_to_document`和`all_chunks`的映射数据打包成一个字典，并使用`numpy.save`保存为`.npy`文件。

### 2. 问答流程 (检索与生成)

当用户输入问题后，系统执行以下流程：

1.  **检索 (`retrieve` 方法)**:
    -   将用户的查询文本转换为查询向量。
    -   在Faiss索引中进行搜索，找出与查询向量最相似的 `k` 个**文本块**的索引。
    -   **关键步骤**：
        -   根据检索到的块索引，从`all_chunks`中获取块的具体内容。
        -   使用`chunks_to_document`映射，找到这些块所属的原始文档ID，并去重。
        -   从原始`documents`列表中获取完整的文档内容。
    -   此方法最终返回两个重要的信息：一组去重后的**原始文档**和一组按相关性排序的**文本块**。

2.  **生成 (`generate_answer` 方法)**:
    -   **构建高级上下文**: 这是与基础版RAG最大的不同。上下文不再是简单地拼接文档，而是分层次地组合信息：
        -   首先，包含检索到的**完整原始文档**，这为LLM提供了宏观、全面的背景知识。
        -   然后，明确地列出**高度相关的文本块**，这为LLM提供了解决问题的精确、直接的线索。
    -   **调用LLM**: 将这个结构化的、信息丰富的上下文连同用户问题一起构建成最终的Prompt，发送给`glm-4-plus`模型。
    -   由于上下文质量更高、针对性更强，LLM能够生成更准确、更深入的回答。 