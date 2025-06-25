# -*- coding: utf-8 -*-
"""
该脚本实现了一个功能完备的、基于文件加载的检索增强生成（RAG）系统。

这是系列项目的最终版本，它在前一个版本的基础上增加了从本地目录动态加载多种格式
文档（如 .txt, .pdf, .docx, .md 等）的功能。系统会自动处理文件读取、文本清理、
文档分块、向量索引和上下文生成，最终实现一个可以基于本地知识库进行问答的完整流程。
"""
import os
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 导入自定义的文件加载工具
from file_utils import load_documents_from_directory, get_default_documents

# --- 全局配置 ---
# 设置TOKENIZERS_PARALLELISM以避免Hugging Face Tokenizers库的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 文档、模型和索引的路径配置
DOCS_DIRECTORY = "./docs"
LOCAL_MODEL_PATH = 'local_m3e_model'
MODEL_NAME = 'moka-ai/m3e-base'
INDEX_FILE_PATH = "m3e_faiss_file_index.bin"
CHUNKS_MAP_PATH = "chunks_file_mapping.npy"

# LLM 和 分块参数配置
LLM_MODEL_NAME = "glm-4-plus"
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 100


class RAGSystemFromFile:
    """
    一个从文件系统加载知识库，并实现完整RAG流程的系统类。
    """
    def __init__(self, doc_directory: str):
        """
        初始化RAG系统。
        参数:
            doc_directory (str): 存放知识库文档的目录路径。
        """
        print("--- RAG系统初始化开始 ---")
        
        # 步骤1: 加载文档
        self.documents, self.doc_sources = self._load_documents(doc_directory)
        
        # 步骤2: 加载嵌入模型
        self.model = self._load_model()
        
        # 初始化用于存储分块和映射关系的数据结构
        self.document_to_chunks: Dict[int, List[int]] = {}
        self.chunks_to_document: Dict[int, int] = {}
        self.all_chunks: List[str] = []
        
        # 步骤3: 创建或加载Faiss索引及文档块的映射关系
        self.index = self._create_or_load_index_and_mappings()
        
        # 步骤4: 初始化大模型客户端
        self.client = self._initialize_openai_client()
        print("--- RAG系统初始化完成 ---\n")

    def _load_documents(self, directory_path: str) -> Tuple[List[str], List[str]]:
        """步骤1: 从指定目录加载所有支持的文档。"""
        print(f"步骤1: 开始从 '{directory_path}' 目录加载文档...")
        documents, sources, errors = load_documents_from_directory(directory_path)

        if errors:
            print("\n  加载过程中出现以下错误或警告:")
            for error in errors:
                print(f"    - {error}")

        if not documents:
            print(f"\n  警告: 在 '{directory_path}' 中未找到任何可处理的文档。")
            print("  将使用内置的默认示例文档继续。")
            documents, sources = get_default_documents()

        print(f"\n  成功加载 {len(documents)} 个文档:")
        for i, source in enumerate(sources):
            print(f"    - [文档{i}] 来源: {source}")
        
        print("\n文档加载完成。\n")
        return documents, sources

    def _load_model(self) -> SentenceTransformer:
        """步骤2: 加载嵌入模型。"""
        print("步骤2: 开始加载嵌入模型...")
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"  从本地路径加载模型: {LOCAL_MODEL_PATH}")
            model = SentenceTransformer(LOCAL_MODEL_PATH)
        else:
            print(f"  本地模型不存在，从网络下载: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            print(f"  保存模型到本地: {LOCAL_MODEL_PATH}")
            model.save(LOCAL_MODEL_PATH)
        print("  模型加载成功！\n")
        return model

    def _chunk_document(self, text: str) -> List[str]:
        """
        将长文本切分成较小的、带有重叠的块，并尝试在句子边界切分以保持语义完整。
        """
        if len(text) <= CHUNK_MAX_CHARS:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_MAX_CHARS, len(text))
            
            # 如果不是最后一块，尝试寻找一个更好的切分点
            if end < len(text):
                # 寻找最后的句号、问号、感叹号等
                sentence_ends = list(re.finditer(r'[。！？.!?]\s*', text[start:end]))
                if sentence_ends:
                    # 在最后一个完整句子处切分
                    end = start + sentence_ends[-1].end()
                else:
                    # 如果没有完整句子，则在最后一个空格处断开
                    last_space = text.rfind(' ', start, end)
                    if last_space > start: # 确保切分点有效
                        end = last_space + 1

            chunks.append(text[start:end])
            
            # 计算下一次的起始点，包含重叠部分
            next_start = end - CHUNK_OVERLAP
            # 必须确保start指针是前进的，避免死循环
            if next_start <= start:
                start += 1
            else:
                start = next_start
        
        return chunks

    def _create_or_load_index_and_mappings(self) -> faiss.Index:
        """
        步骤3: 创建或加载Faiss索引及映射关系。
        注意：当前实现假设如果索引文件存在，则文档内容没有变更。
        在生产环境中，需要更复杂的机制（如文件哈希校验）来检测文件变化并触发重建。
        """
        print("步骤3: 开始创建或加载索引...")
        if os.path.exists(INDEX_FILE_PATH) and os.path.exists(CHUNKS_MAP_PATH):
            print(f"  从本地加载索引和映射文件: {INDEX_FILE_PATH}, {CHUNKS_MAP_PATH}")
            index = faiss.read_index(INDEX_FILE_PATH)
            mapping_data = np.load(CHUNKS_MAP_PATH, allow_pickle=True).item()
            self.document_to_chunks = mapping_data['doc_to_chunks']
            self.chunks_to_document = mapping_data['chunks_to_doc']
            self.all_chunks = mapping_data['all_chunks']
            # 从文件中加载文档来源，以确保一致性
            self.doc_sources = mapping_data.get('doc_sources', self.doc_sources)
            print("  索引和映射加载成功。\n")
            return index
        
        print("  本地索引不完整或不存在，开始创建新索引...")
        
        # 3.1 对所有文档进行分块，并建立映射关系
        print("  3.1 正在对所有已加载文档进行分块...")
        for doc_id, doc_content in enumerate(self.documents):
            chunks = self._chunk_document(doc_content)
            self.document_to_chunks[doc_id] = []
            for chunk in chunks:
                chunk_id = len(self.all_chunks)
                self.all_chunks.append(chunk)
                self.document_to_chunks[doc_id].append(chunk_id)
                self.chunks_to_document[chunk_id] = doc_id
        print(f"  文档分块完成，共生成 {len(self.all_chunks)} 个文本块。")

        # 3.2 为所有文本块生成嵌入向量
        print("  3.2 正在为所有文本块生成嵌入向量...")
        chunk_embeddings = self.model.encode(self.all_chunks, normalize_embeddings=True, show_progress_bar=True)
        chunk_embeddings = np.array(chunk_embeddings).astype('float32')
        print("  嵌入向量生成完成。")

        # 3.3 初始化并构建FAISS索引
        print("  3.3 正在构建FAISS索引...")
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)
        print("  FAISS索引构建完成。")

        # 3.4 保存索引和映射关系到本地
        print(f"  3.4 正在保存索引和映射文件...")
        faiss.write_index(index, INDEX_FILE_PATH)
        mapping_data = {
            'doc_to_chunks': self.document_to_chunks,
            'chunks_to_doc': self.chunks_to_document,
            'all_chunks': self.all_chunks,
            'doc_sources': self.doc_sources
        }
        np.save(CHUNKS_MAP_PATH, mapping_data)
        print(f"  索引和映射已成功保存到: {INDEX_FILE_PATH}, {CHUNKS_MAP_PATH}\n")
        return index

    def _initialize_openai_client(self) -> OpenAI:
        """步骤4: 初始化大模型客户端。"""
        print("步骤4: 初始化大模型客户端...")
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("错误: ZHIPUAI_API_KEY 环境变量未设置。请先设置该变量。")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        print("  大模型客户端初始化成功。\n")
        return client

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Tuple[str, str]], List[Tuple[int, str, str]]]:
        """
        步骤5: 检索。根据用户查询，找到最相关的原始文档和文本块。
        返回:
            - 相关原始文档列表: [(文档内容, 文档来源), ...]
            - 相关文本块列表: [(文档ID, 块内容, 文档来源), ...]
        """
        print(f"步骤5: 开始检索，查询: '{query}'")
        # 5.1 为查询生成嵌入向量
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # 5.2 在FAISS索引中搜索最相似的k个文本块
        _, chunk_indices = self.index.search(query_embedding, k)
        
        # 5.3 根据块索引，获取块内容和原始文档ID及来源
        retrieved_doc_ids = set()
        retrieved_chunks_with_source = []
        print("\n  检索到的相关文本块:")
        for chunk_idx in chunk_indices[0]:
            if 0 <= chunk_idx < len(self.all_chunks):
                doc_id = self.chunks_to_document[int(chunk_idx)]
                retrieved_doc_ids.add(doc_id)
                chunk_content = self.all_chunks[int(chunk_idx)]
                source_file = self.doc_sources[doc_id]
                retrieved_chunks_with_source.append((doc_id, chunk_content, source_file))
                print(f"    [来自: {os.path.basename(source_file)}] {chunk_content[:80]}...")

        # 5.4 根据文档ID获取去重后的原始文档内容和来源
        retrieved_docs_with_source = sorted([(self.documents[doc_id], self.doc_sources[doc_id]) for doc_id in retrieved_doc_ids])
        
        print("\n  将用于上下文的原始文档:")
        for doc, source in retrieved_docs_with_source:
            print(f"    - 来源: {source}, 内容预览: {doc[:80]}...")
        
        print("\n检索完成。\n")
        return retrieved_docs_with_source, retrieved_chunks_with_source

    def generate_answer(self, query: str, retrieved_docs: List[Tuple[str, str]], retrieved_chunks: List[Tuple[int, str, str]]) -> str:
        """
        步骤6: 生成。将检索到的信息整合为上下文，调用大模型生成答案。
        """
        print("步骤6: 开始生成答案...")
        # 6.1 构建上下文，清晰地包含文档来源和文本块来源
        context = "请严格依据以下提供的上下文信息来回答问题，不要使用任何外部知识或进行推断。\n\n"
        
        context += "--- 相关原始文档 ---\n"
        for i, (doc, source) in enumerate(retrieved_docs):
            context += f"【文档{i+1} | 来源: {os.path.basename(source)}】\n{doc}\n\n"
        
        context += "\n--- 高度相关的文本块 ---\n"
        for i, (doc_id, chunk, source) in enumerate(retrieved_chunks):
            context += f"【文本块{i+1} | 来源: {os.path.basename(source)}】\n{chunk}\n\n"
        
        # 6.2 构建最终的prompt
        prompt = f"上下文信息:\n---\n{context}---\n\n问题: {query}\n\n请根据以上上下文信息，回答问题。"
        
        # 6.3 调用大模型
        print("  正在调用大模型生成答案...")
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手。你的任务是严格、忠实地依据用户提供的上下文信息来回答问题。"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        print("  答案生成完成。\n")
        return answer

def main():
    """
    主函数，演示整个RAG流程。
    """
    # 初始化RAG系统，它会自动从DOCS_DIRECTORY加载文件并完成所有准备工作
    rag_system = RAGSystemFromFile(DOCS_DIRECTORY)

    # --- 开始问答流程 ---
    query = "界面新闻公司的简介，100字左右"
    print(f"--- 开始问答流程，用户查询: \"{query}\" ---\n")
    
    # 步骤5: 检索相关文档和文本块
    retrieved_docs, retrieved_chunks = rag_system.retrieve(query)
    
    # 步骤6: 基于检索到的信息生成答案
    answer = rag_system.generate_answer(query, retrieved_docs, retrieved_chunks)
    
    # --- 最终结果 ---
    print("\n--- 最终结果 ---")
    print(f"查询: {query}")
    print(f"回答: \n{answer}")

if __name__ == "__main__":
    main()