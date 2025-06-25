# -*- coding: utf-8 -*-
"""
该脚本实现了一个带有文档分块功能的高级检索增强生成（RAG）系统。

与基础版RAG不同，此系统首先将长文档切分成更小的、有重叠的块（chunks）。
然后对这些块进行向量化和索引。当用户提问时，系统检索出最相关的文本块，
再根据这些块追溯到原始文档。最后，将检索到的原始文档和最相关的文本块
一起作为上下文，交给大型语言模型生成更精确、更聚焦的答案。
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
import re
from typing import List, Dict, Tuple

# 设置TOKENIZERS_PARALLELISM以避免Hugging Face Tokenizers库的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 全局配置 ---
LOCAL_MODEL_PATH = 'local_m3e_model'
MODEL_NAME = 'moka-ai/m3e-base'
INDEX_FILE_PATH = "m3e_faiss_chunk_index.bin"
CHUNKS_MAP_PATH = "chunks_mapping_data.npy"
LLM_MODEL_NAME = "glm-4-plus"
CHUNK_MAX_CHARS = 500  # 每个块的最大字符数
CHUNK_OVERLAP = 100    # 相邻块之间的重叠字符数


class RAGSystemWithChunking:
    """
    一个封装了文档分块、索引、检索和生成全流程的RAG系统类。
    """
    def __init__(self, documents: List[str]):
        """
        初始化RAG系统。
        参数:
            documents (List[str]): 需要被索引的原始文档列表。
        """
        print("--- RAG系统初始化开始 ---")
        self.documents = documents
        self.model = self._load_model()
        
        # 初始化用于存储分块和映射关系的数据结构
        self.document_to_chunks: Dict[int, List[int]] = {}
        self.chunks_to_document: Dict[int, int] = {}
        self.all_chunks: List[str] = []
        
        self.index = self._create_or_load_index_and_mappings()
        self.client = self._initialize_openai_client()
        print("--- RAG系统初始化完成 ---\n")

    def _load_model(self) -> SentenceTransformer:
        """步骤1: 加载嵌入模型。优先从本地加载，否则从网络下载。"""
        print("步骤1: 开始加载嵌入模型...")
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"  从本地路径加载模型: {LOCAL_MODEL_PATH}")
            model = SentenceTransformer(LOCAL_MODEL_PATH)
        else:
            print(f"  本地模型不存在，从网络下载: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            print(f"  保存模型到本地: {LOCAL_MODEL_PATH}")
            model.save(LOCAL_MODEL_PATH)
        print("模型加载成功！\n")
        return model

    def _chunk_document(self, text: str) -> List[str]:
        """
        将长文本切分成较小的、带有重叠的块，以保证上下文的连续性。
        """
        if len(text) <= CHUNK_MAX_CHARS:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_MAX_CHARS
            current_chunk = text[start:end]
            chunks.append(current_chunk)
            start += CHUNK_MAX_CHARS - CHUNK_OVERLAP
        return chunks

    def _create_or_load_index_and_mappings(self) -> faiss.Index:
        """步骤2: 创建或加载Faiss索引及文档块的映射关系。"""
        print("步骤2: 开始创建或加载索引...")
        if os.path.exists(INDEX_FILE_PATH) and os.path.exists(CHUNKS_MAP_PATH):
            print(f"  从本地加载索引和映射文件: {INDEX_FILE_PATH}, {CHUNKS_MAP_PATH}")
            index = faiss.read_index(INDEX_FILE_PATH)
            mapping_data = np.load(CHUNKS_MAP_PATH, allow_pickle=True).item()
            self.document_to_chunks = mapping_data['doc_to_chunks']
            self.chunks_to_document = mapping_data['chunks_to_doc']
            self.all_chunks = mapping_data['all_chunks']
            print("  索引和映射加载成功。\n")
            return index
        else:
            print("  本地索引不存在，开始创建新索引...")
            
            # 2.1 文档分块并建立映射关系
            print("  2.1 正在对文档进行分块...")
            for doc_id, doc in enumerate(self.documents):
                chunks = self._chunk_document(doc)
                self.document_to_chunks[doc_id] = []
                for chunk in chunks:
                    chunk_id = len(self.all_chunks)
                    self.all_chunks.append(chunk)
                    self.document_to_chunks[doc_id].append(chunk_id)
                    self.chunks_to_document[chunk_id] = doc_id
            print(f"  文档分块完成，共生成 {len(self.all_chunks)} 个文本块。")

            # 2.2 为所有文本块生成嵌入向量
            print("  2.2 正在为所有文本块生成嵌入向量...")
            chunk_embeddings = self.model.encode(self.all_chunks, normalize_embeddings=True)
            chunk_embeddings = np.array(chunk_embeddings).astype('float32')
            print("  嵌入向量生成完成。")

            # 2.3 初始化并构建FAISS索引
            print("  2.3 正在构建FAISS索引...")
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(chunk_embeddings)
            print("  FAISS索引构建完成。")

            # 2.4 保存索引和映射关系到本地
            print(f"  2.4 正在保存索引和映射文件...")
            faiss.write_index(index, INDEX_FILE_PATH)
            mapping_data = {
                'doc_to_chunks': self.document_to_chunks,
                'chunks_to_doc': self.chunks_to_document,
                'all_chunks': self.all_chunks
            }
            np.save(CHUNKS_MAP_PATH, mapping_data)
            print(f"  索引和映射已成功保存到: {INDEX_FILE_PATH}, {CHUNKS_MAP_PATH}\n")
            return index

    def _initialize_openai_client(self) -> OpenAI:
        """步骤3: 初始化大模型客户端。"""
        print("步骤3: 初始化大模型客户端...")
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("错误: ZHIPUAI_API_KEY 环境变量未设置。请先设置该变量。")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        print("  大模型客户端初始化成功。\n")
        return client

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[Tuple[int, str]]]:
        """
        步骤4: 检索。根据用户查询，找到最相关的原始文档和文本块。
        参数:
            query (str): 用户的查询。
            k (int): 需要检索的相关文本块数量。
        返回:
            一个元组，包含:
            - List[str]: 去重后的相关原始文档列表。
            - List[Tuple[int, str]]: 相关的文本块列表，每个元素是(文档ID, 块内容)。
        """
        print(f"步骤4: 开始检索，查询: '{query}'")
        # 4.1 为查询生成嵌入向量
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # 4.2 在FAISS索引中搜索最相似的k个文本块
        _, chunk_indices = self.index.search(query_embedding, k)
        
        # 4.3 根据块索引，获取块内容和原始文档ID
        retrieved_doc_ids = set()
        retrieved_chunks = []
        print("\n  检索到的相关文本块:")
        for chunk_idx in chunk_indices[0]:
            if 0 <= chunk_idx < len(self.all_chunks):
                doc_id = self.chunks_to_document[int(chunk_idx)]
                retrieved_doc_ids.add(doc_id)
                chunk_content = self.all_chunks[int(chunk_idx)]
                retrieved_chunks.append((doc_id, chunk_content))
                print(f"    [来自文档{doc_id}] {chunk_content}")

        # 4.4 根据文档ID获取去重后的原始文档内容
        retrieved_docs = sorted([self.documents[doc_id] for doc_id in retrieved_doc_ids])
        print("\n  将用于上下文的原始文档:")
        for doc in retrieved_docs:
            print(f"    - {doc[:100]}...") # 打印部分内容
        
        print("\n检索完成。\n")
        return retrieved_docs, retrieved_chunks

    def generate_answer(self, query: str, retrieved_docs: List[str], retrieved_chunks: List[Tuple[int, str]]) -> str:
        """
        步骤5: 生成。将检索到的信息整合为上下文，调用大模型生成答案。
        """
        print("步骤5: 开始生成答案...")
        # 5.1 构建上下文，包含检索到的原始文档和更精确的相关文本块
        context = "请仅基于以下提供的上下文信息回答问题。\n\n"
        context += "--- 相关原始文档 ---\n"
        for doc in retrieved_docs:
            context += doc + "\n"
        
        context += "\n--- 高度相关的文本块 ---\n"
        for doc_id, chunk in retrieved_chunks:
            context += f"[来自文档{doc_id}] {chunk}\n"
        
        # 5.2 构建最终的prompt
        prompt = f"上下文信息:\n{context}\n\n问题: {query}\n\n请根据以上上下文信息，回答问题。"
        
        # 5.3 调用大模型
        print("  正在调用大模型生成答案...")
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手。你的任务是严格依据用户提供的上下文信息来回答问题，不要使用任何外部知识或进行推断。"},
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
    # --- 知识库数据 ---
    documents = [
        "界面新闻是中国具有影响力的原创财经新媒体，由上海报业集团出品，2014年9月创立。界面新闻客户端曾被中央网信办评为 App影响力十佳。2017—2022年位居艾瑞商业资讯类移动App指数第一名。",
        "企业价值观：真实准确、客观公正、有担当。Slogan：界面新闻，只服务于独立思考的人群。",
        "创始人：何力，毕业于首都师范大学，2014年参与创立界面新闻并担任CEO。何力先生拥有丰富的媒体行业经验，在创立界面新闻之前，他曾担任《经济观察报》社长、总编辑，以及《第一财经周刊》的总编辑。他的愿景是打造一个服务于中国新一代商业领袖和决策者的财经媒体平台。"
    ]

    # 初始化RAG系统（包含加载模型、分块、建索引等所有准备工作）
    rag_system = RAGSystemWithChunking(documents)

    # --- 开始问答流程 ---
    query = "界面新闻创始人的背景是什么？"
    print(f"--- 开始问答流程，用户查询: \"{query}\" ---\n")
    
    # 步骤4: 检索相关文档和文本块
    retrieved_docs, retrieved_chunks = rag_system.retrieve(query)
    
    # 步骤5: 基于检索到的信息生成答案
    answer = rag_system.generate_answer(query, retrieved_docs, retrieved_chunks)
    
    # --- 最终结果 ---
    print("--- 最终结果 ---")
    print(f"查询: {query}")
    print(f"回答: \n{answer}")

if __name__ == "__main__":
    main()