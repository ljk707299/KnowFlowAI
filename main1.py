# -*- coding: utf-8 -*-
"""
该脚本实现了一个基本的检索增强生成（RAG）系统。

该系统使用一个句子转换器模型（m3e-base）为一组文档生成嵌入。
然后使用FAISS对这些嵌入进行索引，以实现高效的相似性搜索。
当给出查询时，系统从索引中检索最相关的文档，
并使用大型语言模型（如GLM-4）根据查询和检索到的上下文生成答案。
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
from typing import List

# 设置TOKENIZERS_PARALLELISM为false以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 配置 ---
# 存储本地句子转换器模型的路径
LOCAL_MODEL_PATH = 'local_m3e_model'
# Hugging Face上的句子转换器模型的名称
MODEL_NAME = 'moka-ai/m3e-base'
# 存储FAISS索引文件的路径
INDEX_FILE_PATH = "m3e_faiss_index.bin"
# 用于生成的LLM模型
LLM_MODEL_NAME = "glm-4-plus"


class RAGSystem:
    """
    一个封装了整个检索增强生成系统的类。
    """

    def __init__(self, documents: List[str]):
        """
        初始化RAG系统。

        参数:
            documents (List[str]): 需要被索引的文档列表。
        """
        self.documents = documents
        # 加载嵌入模型
        self.model = self._load_model()
        # 创建或加载文档索引
        self.index = self._create_or_load_index()
        # 初始化LLM客户端
        self.client = self._initialize_openai_client()

    def _load_model(self) -> SentenceTransformer:
        """
        加载句子转换器模型。
        它会首先检查本地副本，如果不存在则从Hugging Face下载。

        返回:
            SentenceTransformer: 加载的句子转换器模型。
        """
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"从本地路径加载模型: {LOCAL_MODEL_PATH}")
            return SentenceTransformer(LOCAL_MODEL_PATH)
        else:
            print(f"未找到本地模型。正在从Hugging Face下载: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            print(f"正在将模型保存到本地路径: {LOCAL_MODEL_PATH}")
            model.save(LOCAL_MODEL_PATH)
            return model

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        为文本列表生成嵌入。

        参数:
            texts (List[str]): 需要嵌入的文本列表。

        返回:
            np.ndarray: 嵌入的numpy数组。
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    def _create_or_load_index(self) -> faiss.Index:
        """
        创建一个新的FAISS索引或从磁盘加载一个现有的索引。

        返回:
            faiss.Index: FAISS索引。
        """
        if os.path.exists(INDEX_FILE_PATH):
            print(f"从以下位置加载FAISS索引: {INDEX_FILE_PATH}")
            return faiss.read_index(INDEX_FILE_PATH)
        else:
            print("正在创建新的FAISS索引。")
            # 1. 为所有文档生成嵌入
            doc_embeddings = self.get_embeddings(self.documents)
            dimension = doc_embeddings.shape[1]
            # 2. 创建一个L2距离的FAISS索引
            index = faiss.IndexFlatL2(dimension)
            # 3. 将文档嵌入添加到索引中
            index.add(doc_embeddings)
            # 4. 将索引写入磁盘
            faiss.write_index(index, INDEX_FILE_PATH)
            print(f"索引已创建并保存到: {INDEX_FILE_PATH}")
            return index

    def _initialize_openai_client(self) -> OpenAI:
        """
        使用环境变量中的API密钥初始化OpenAI客户端。

        请确保设置ZHIPUAI_API_KEY环境变量。
        
        返回:
            OpenAI: 初始化后的OpenAI客户端。
        """
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("未设置ZHIPUAI_API_KEY环境变量。")
        
        return OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )

    def retrieve_docs(self, query: str, k: int = 3) -> List[str]:
        """
        为给定查询检索前k个最相关的文档。

        参数:
            query (str): 用户的查询。
            k (int): 要检索的文档数量。

        返回:
            List[str]: 检索到的文档列表。
        """
        # 1. 为查询生成嵌入
        query_embedding = self.get_embeddings([query])
        # 2. 在FAISS索引中搜索最相似的文档
        _, indices = self.index.search(query_embedding, k)
        # 3. 返回最相关的文档内容
        return [self.documents[i] for i in indices[0]]

    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str:
        """
        使用LLM根据查询和检索到的文档生成答案。

        参数:
            query (str): 用户的查询。
            retrieved_docs (List[str]): 用作上下文的检索文档列表。

        返回:
            str: 生成的答案。
        """
        # 1. 将检索到的文档拼接成上下文
        context = "\n".join(retrieved_docs)
        # 2. 构建prompt
        prompt = f"上下文:\n{context}\n\n问题: {query}\n答案:"

        # 3. 调用LLM生成答案
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content

def main():
    """
    运行RAG系统的主函数。
    """
    # 步骤1: 定义本地知识库
    # --- 数据 ---
    documents = [
        "界面新闻是中国具有影响力的原创财经新媒体，由上海报业集团出品，2014年9月创立。界面新闻客户端曾被中央网信办评为 App影响力十佳。2017—2022年位居艾瑞商业资讯类移动App指数第一名",
        "企业价值观：真实准确、客观公正、有担当 ，Slogan：界面新闻，只服务于独立思考的人群",
        "创始人： 何力，毕业于首都师范大学，2014年参与创立界面新闻并担任CEO，界面新闻是中国具有影响力的原创财经新媒体，由上海报业集团出品"
    ]

    # 步骤2: 初始化RAG系统
    # 这个过程会自动加载模型、创建或加载索引。
    rag_system = RAGSystem(documents)

    # 步骤3: 定义用户查询
    # --- 测试查询 ---
    query = "界面新闻的创始人是谁？"
    print(f"\n查询: {query}")

    # 步骤4: 检索(Retrieval)
    # 根据查询从知识库中检索最相关的文档。
    retrieved_docs = rag_system.retrieve_docs(query)
    print("\n检索到的文档:")
    for doc in retrieved_docs:
        print(f"- {doc}")

    # 步骤5: 生成(Generation)
    # 将检索到的文档作为上下文，连同原始查询一起提供给LLM生成答案。
    answer = rag_system.generate_answer(query, retrieved_docs)
    print(f"\n答案: {answer}")

if __name__ == "__main__":
    main()
    