# file_utils.py

import os
import glob
import re
import pandas as pd
from bs4 import BeautifulSoup
import markdown

# 尝试导入可能不存在的包
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("警告: python-docx 未安装，无法处理 .docx 文件")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("警告: PyPDF2 未安装，无法处理 .pdf 文件")

def clean_text(text):
    """
    清理文本，去除多余的空格和换行
    """
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 替换多个换行为单个换行
    text = re.sub(r'\n+', '\n', text)
    # 去除首尾空白
    return text.strip()

def load_text_file(file_path):
    """加载文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return clean_text(content), None
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
            return clean_text(content), None
        except Exception as e:
            return None, f"无法读取文本文件 {file_path}: {str(e)}"
    except Exception as e:
        return None, f"读取文本文件 {file_path} 时出错: {str(e)}"

def load_docx_file(file_path):
    """加载Word文档"""
    if not DOCX_AVAILABLE:
        return None, "python-docx 包未安装，无法处理 .docx 文件"
    
    try:
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        content = '\n'.join(paragraphs)
        return clean_text(content), None
    except Exception as e:
        return None, f"读取Word文档 {file_path} 时出错: {str(e)}"

def load_pdf_file(file_path):
    """加载PDF文件"""
    if not PDF_AVAILABLE:
        return None, "PyPDF2 包未安装，无法处理 .pdf 文件"
    
    try:
        content = ''
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page_content = pdf_reader.pages[page_num].extract_text()
                if page_content:  # 有些PDF页面可能提取不出文本
                    content += page_content + '\n'
        
        if not content.strip():
            return None, f"PDF文件 {file_path} 未能提取到文本内容"
        
        return clean_text(content), None
    except Exception as e:
        return None, f"读取PDF文件 {file_path} 时出错: {str(e)}"

def load_markdown_file(file_path):
    """加载Markdown文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 将markdown转换为HTML，然后提取纯文本
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.get_text()
        
        return clean_text(content), None
    except Exception as e:
        return None, f"读取Markdown文件 {file_path} 时出错: {str(e)}"

def load_excel_file(file_path):
    """加载Excel文件"""
    try:
        # 读取所有sheet并合并
        df_list = pd.read_excel(file_path, sheet_name=None)
        content = ''
        
        for sheet_name, df in df_list.items():
            content += f"表格: {sheet_name}\n"
            # 将DataFrame转换为字符串，限制每个单元格的内容长度
            for _, row in df.iterrows():
                for col_name, value in row.items():
                    # 转换为字符串并限制长度
                    str_value = str(value)
                    if len(str_value) > 100:  # 如果单元格内容过长，截断
                        str_value = str_value[:100] + "..."
                    content += f"{col_name}: {str_value}\n"
                content += "---\n"  # 行分隔符
        
        return clean_text(content), None
    except Exception as e:
        return None, f"读取Excel文件 {file_path} 时出错: {str(e)}"

def load_documents_from_directory(directory_path, file_types=None):
    """
    从指定目录加载多种类型的文档
    
    参数:
        directory_path: 文档所在目录路径
        file_types: 文件类型列表，如 ['.txt', '.docx', '.pdf', '.md', '.xlsx']
                   如果为None，则加载所有支持的类型
    
    返回:
        documents: 文档内容列表
        doc_sources: 文档来源信息列表
        errors: 加载过程中的错误信息
    """
    if file_types is None:
        file_types = ['.txt', '.docx', '.pdf', '.md', '.xlsx', '.xls']
    
    documents = []
    doc_sources = []
    errors = []
    
    # 确保目录存在
    if not os.path.exists(directory_path):
        errors.append(f"目录 {directory_path} 不存在")
        return documents, doc_sources, errors
    
    # 遍历所有指定类型的文件
    for file_type in file_types:
        file_paths = glob.glob(os.path.join(directory_path, f'*{file_type}'))
        for file_path in file_paths:
            content = None
            error = None
            
            # 根据文件类型调用不同的加载函数
            if file_path.endswith('.txt'):
                content, error = load_text_file(file_path)
            
            elif file_path.endswith('.docx'):
                content, error = load_docx_file(file_path)
            
            elif file_path.endswith('.pdf'):
                content, error = load_pdf_file(file_path)
            
            elif file_path.endswith('.md'):
                content, error = load_markdown_file(file_path)
            
            elif file_path.endswith(('.xlsx', '.xls')):
                content, error = load_excel_file(file_path)
            
            # 处理加载结果
            if content:
                documents.append(content)
                doc_sources.append(file_path)
                print(f"成功加载文件: {file_path}")
            
            if error:
                errors.append(error)
                print(f"警告: {error}")
    
    # 如果没有成功加载任何文档
    if not documents:
        errors.append(f"在目录 {directory_path} 中未找到任何可处理的文档")
    
    return documents, doc_sources, errors

def get_default_documents():
    """返回默认的示例文档"""
    documents = [
        "界面新闻是中国具有影响力的原创财经新媒体，由上海报业集团出品，2014年9月创立。界面新闻客户端曾被中央网信办评为 App影响力十佳。2017—2022年位居艾瑞商业资讯类移动App指数第一名",
        "企业价值观：真实准确、客观公正、有担当 ，Slogan：界面新闻，只服务于独立思考的人群",
        "创始人： 何力，毕业于首都师范大学，2014年参与创立界面新闻并担任CEO，界面新闻是中国具有影响力的原创财经新媒体，由上海报业集团出品"
    ]
    doc_sources = ["默认样例1", "默认样例2", "默认样例3"]
    return documents, doc_sources

def create_example_files(directory_path):
    """创建示例文件"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # 创建示例文本文件
    with open(os.path.join(directory_path, "example.txt"), "w", encoding="utf-8") as f:
        f.write("这是一个示例文档，用于测试文档加载功能，界面新闻是中国具有影响力的原创财经新媒体，由上海报业集团出品，2014年9月创立。界面新闻客户端曾被中央网信办评为App影响力十佳。2017—2022年位居艾瑞商业资讯类移动App指数第一名")
    
    print(f"已创建示例文档: {os.path.join(directory_path, 'example.txt')}")
    return True

# 演示用法
if __name__ == "__main__":
    # 设置文档目录
    doc_dir = "docs"
    
    # 创建示例文件
    if not os.path.exists(doc_dir):
        create_example_files(doc_dir)
    
    # 加载文档
    docs, sources, errors = load_documents_from_directory(doc_dir)
    
    # 显示结果
    print(f"\n成功加载 {len(docs)} 个文档:")
    for i, (doc, source) in enumerate(zip(docs, sources)):
        print(f"\n文档 {i+1} 来源: {source}")
        # 只显示文档的开头部分
        preview = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"内容预览: {preview}")
    
    # 如果有错误，显示错误信息
    if errors:
        print("\n加载过程中出现以下错误:")
        for error in errors:
            print(f"- {error}")
    
    # 如果没有加载到文档，使用默认样例
    if not docs:
        docs, sources = get_default_documents()
        print("\n使用默认样例文档:")
        for i, (doc, source) in enumerate(zip(docs, sources)):
            print(f"\n样例 {i+1} 来源: {source}")
            print(f"内容: {doc}")