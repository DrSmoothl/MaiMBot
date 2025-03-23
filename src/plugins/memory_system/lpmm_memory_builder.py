import datetime
import time
import random
import jieba
import networkx as nx
import numpy as np
import faiss
import os
import json
import hashlib
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import sys
import os
from dataclasses import dataclass
import asyncio

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.common.database import db
from src.common.logger import get_module_logger
from src.plugins.chat.utils import get_closest_chat_from_db
from src.plugins.memory_system.memory import Memory_graph, Hippocampus, cosine_similarity, calculate_information_content, LLM_request
from src.plugins.chat.config import global_config

logger = get_module_logger("lpmm_memory_builder")

# 定义命名空间常量
ENT_NAMESPACE = "entity"
REL_NAMESPACE = "relation"
PG_NAMESPACE = "paragraph"
RAG_GRAPH_NAMESPACE = "rag_graph"
RAG_ENT_CNT_NAMESPACE = "rag_ent_cnt"
RAG_PG_HASH_NAMESPACE = "rag_pg_hash"

@dataclass
class EmbeddingItem:
    """嵌入存储项"""
    
    hash_key: str
    embedding: List[float]
    content: str
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "hash_key": self.hash_key,
            "embedding": self.embedding,
            "content": self.content
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'EmbeddingItem':
        """从字典创建实例"""
        return EmbeddingItem(
            hash_key=data["hash_key"],
            embedding=data["embedding"],
            content=data["content"]
        )
    
    @staticmethod
    def calculate_similarity(item1: Union[List[float], 'EmbeddingItem'], 
                            item2: Union[List[float], 'EmbeddingItem']) -> float:
        """计算两个嵌入向量的相似度"""
        # 提取嵌入向量
        if isinstance(item1, EmbeddingItem):
            embed1 = np.array(item1.embedding)
        else:
            embed1 = np.array(item1)
            
        if isinstance(item2, EmbeddingItem):
            embed2 = np.array(item2.embedding)
        else:
            embed2 = np.array(item2)
            
        # 计算余弦相似度
        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(embed1, embed2) / (norm1 * norm2)


class EnhancedEmbeddingStore:
    """增强型嵌入存储类，支持持久化和FAISS索引"""
    
    def __init__(self, namespace: str, data_dir: str = "./data/embeddings"):
        self.namespace = namespace
        self.data_dir = data_dir
        self.store: Dict[str, EmbeddingItem] = {}
        self.faiss_index = None
        self.hash_to_idx: Dict[str, int] = {}  # 哈希值到索引的映射
        self.idx_to_hash: Dict[int, str] = {}  # 索引到哈希值的映射
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _get_file_paths(self) -> Tuple[str, str, str]:
        """获取文件路径"""
        store_path = os.path.join(self.data_dir, f"{self.namespace}_store.json")
        index_path = os.path.join(self.data_dir, f"{self.namespace}_index.bin")
        mapping_path = os.path.join(self.data_dir, f"{self.namespace}_mapping.json")
        return store_path, index_path, mapping_path
    
    def add_item(self, content: str, embedding: List[float]) -> str:
        """添加一个项到存储"""
        # 创建哈希键
        hash_key = f"{self.namespace}-{self._get_md5(content)}"
        
        # 如果已经存在，直接返回
        if hash_key in self.store:
            return hash_key
            
        # 创建并存储
        item = EmbeddingItem(hash_key=hash_key, embedding=embedding, content=content)
        self.store[hash_key] = item
        
        # 标记索引需要重建
        self.faiss_index = None
        
        return hash_key
    
    def batch_add_items(self, contents: List[str], embeddings: List[List[float]]) -> List[str]:
        """批量添加项"""
        if len(contents) != len(embeddings):
            raise ValueError("内容和嵌入数量不匹配")
            
        hash_keys = []
        for content, embedding in zip(contents, embeddings):
            hash_key = self.add_item(content, embedding)
            hash_keys.append(hash_key)
            
        return hash_keys
    
    def get_item(self, hash_key: str) -> Optional[EmbeddingItem]:
        """获取一个项"""
        return self.store.get(hash_key)
    
    def get_all_hash_keys(self) -> List[str]:
        """获取所有哈希键"""
        return list(self.store.keys())
    
    def build_faiss_index(self) -> None:
        """构建FAISS索引"""
        if not self.store:
            logger.warning(f"存储为空，无法构建{self.namespace}的FAISS索引")
            return
            
        # 准备数据
        embeddings = []
        self.hash_to_idx = {}
        self.idx_to_hash = {}
        
        for idx, (hash_key, item) in enumerate(self.store.items()):
            embeddings.append(item.embedding)
            self.hash_to_idx[hash_key] = idx
            self.idx_to_hash[idx] = hash_key
            
        # 转换为numpy数组
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # L2归一化
        faiss.normalize_L2(embeddings_np)
        
        # 创建索引
        dimension = len(embeddings[0]) if embeddings else 768  # 默认维度
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积（即余弦相似度，因为已经归一化）
        self.faiss_index.add(embeddings_np)
        
        logger.info(f"已构建{self.namespace}的FAISS索引，包含{len(embeddings)}个项")
    
    def search(self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """搜索最相似的向量"""
        if not self.store:
            logger.warning(f"{self.namespace}存储为空，无法执行搜索")
            return []
            
        # 确保FAISS索引已构建
        if self.faiss_index is None:
            self.build_faiss_index()
            
        if self.faiss_index is None:
            logger.warning(f"无法构建{self.namespace}的FAISS索引")
            return []
            
        # 准备查询向量
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # 执行搜索
        distances, indices = self.faiss_index.search(query_np, top_k)
        
        # 处理结果
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.idx_to_hash):
                continue
                
            hash_key = self.idx_to_hash[idx]
            if score >= threshold:
                results.append((hash_key, float(score)))
                
        return results
    
    def save(self) -> None:
        """保存到文件"""
        store_path, index_path, mapping_path = self._get_file_paths()
        
        # 保存存储
        with open(store_path, 'w', encoding='utf-8') as f:
            store_data = {hash_key: item.to_dict() for hash_key, item in self.store.items()}
            json.dump(store_data, f, ensure_ascii=False)
            
        # 保存FAISS索引
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, index_path)
            
            # 保存映射
            with open(mapping_path, 'w', encoding='utf-8') as f:
                mapping_data = {
                    "hash_to_idx": {k: int(v) for k, v in self.hash_to_idx.items()},
                    "idx_to_hash": {int(k): v for k, v in self.idx_to_hash.items()}
                }
                json.dump(mapping_data, f, ensure_ascii=False)
                
        logger.info(f"已保存{self.namespace}存储，共{len(self.store)}个项")
    
    def load(self) -> bool:
        """从文件加载"""
        store_path, index_path, mapping_path = self._get_file_paths()
        
        # 检查文件是否存在
        if not os.path.exists(store_path):
            logger.warning(f"{self.namespace}存储文件不存在: {store_path}")
            return False
            
        # 加载存储
        try:
            with open(store_path, 'r', encoding='utf-8') as f:
                store_data = json.load(f)
                self.store = {hash_key: EmbeddingItem.from_dict(item_data) 
                             for hash_key, item_data in store_data.items()}
                
            # 加载FAISS索引和映射
            if os.path.exists(index_path) and os.path.exists(mapping_path):
                self.faiss_index = faiss.read_index(index_path)
                
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    self.hash_to_idx = {k: int(v) for k, v in mapping_data["hash_to_idx"].items()}
                    self.idx_to_hash = {int(k): v for k, v in mapping_data["idx_to_hash"].items()}
            else:
                # 如果索引或映射不存在，重新构建
                self.build_faiss_index()
                
            logger.info(f"已加载{self.namespace}存储，共{len(self.store)}个项")
            return True
        except Exception as e:
            logger.error(f"加载{self.namespace}存储失败: {str(e)}")
            return False
    
    def _get_md5(self, text: str) -> str:
        """计算文本的MD5哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class LPMMMemoryBuilder:
    """结合LPMM的记忆构建器"""
    
    def __init__(self, data_dir: str = "./data/lpmm", llm_client: Any = None):
        self.data_dir = data_dir
        self.llm_client = llm_client
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 创建嵌入存储
        self.paragraph_store = EnhancedEmbeddingStore(PG_NAMESPACE, self.data_dir)
        self.entity_store = EnhancedEmbeddingStore(ENT_NAMESPACE, self.data_dir)
        self.relation_store = EnhancedEmbeddingStore(REL_NAMESPACE, self.data_dir)
        
        # 知识图谱
        self.knowledge_graph = nx.DiGraph()
        
        # 加载存储的数据
        self._load_data()
        
    def _load_data(self) -> None:
        """加载已存储的数据"""
        # 加载嵌入存储
        self.paragraph_store.load()
        self.entity_store.load()
        self.relation_store.load()
        
        # 加载知识图谱
        graph_path = os.path.join(self.data_dir, "knowledge_graph.json")
        if os.path.exists(graph_path):
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    self.knowledge_graph = nx.node_link_graph(graph_data)
                logger.info(f"已加载知识图谱，包含{self.knowledge_graph.number_of_nodes()}个节点和{self.knowledge_graph.number_of_edges()}条边")
            except Exception as e:
                logger.error(f"加载知识图谱失败: {str(e)}")
                self.knowledge_graph = nx.DiGraph()
    
    def _save_data(self) -> None:
        """保存数据"""
        # 保存嵌入存储
        self.paragraph_store.save()
        self.entity_store.save()
        self.relation_store.save()
        
        # 保存知识图谱
        graph_path = os.path.join(self.data_dir, "knowledge_graph.json")
        try:
            with open(graph_path, 'w', encoding='utf-8') as f:
                graph_data = nx.node_link_data(self.knowledge_graph)
                json.dump(graph_data, f, ensure_ascii=False)
            logger.info(f"已保存知识图谱，包含{self.knowledge_graph.number_of_nodes()}个节点和{self.knowledge_graph.number_of_edges()}条边")
        except Exception as e:
            logger.error(f"保存知识图谱失败: {str(e)}")
    
    async def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """从文本中提取知识三元组"""
        if not self.llm_client:
            logger.warning("LLM客户端未初始化，无法提取三元组")
            return []
            
        # 使用LLM提取三元组
        prompt = f"""请从以下文本中提取重要的知识三元组(主体, 关系, 客体)。
输出格式应该为JSON数组，每个元素包含三个字符串: [主体, 关系, 客体]
仅返回JSON数组，不要有其他文字。

文本: {text}"""

        try:
            response, _, _ = await self.llm_client.generate_response(prompt)
            
            # 解析返回的JSON
            triples_data = json.loads(response)
            
            # 验证格式
            validated_triples = []
            for triple in triples_data:
                if (isinstance(triple, list) and len(triple) == 3 and
                    all(isinstance(item, str) for item in triple)):
                    validated_triples.append((triple[0], triple[1], triple[2]))
                    
            return validated_triples
        except Exception as e:
            logger.error(f"提取三元组失败: {str(e)}")
            return []
    
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        if not self.llm_client:
            logger.warning("LLM客户端未初始化，无法获取嵌入向量")
            return []
            
        try:
            return await self.llm_client.send_embedding_request(text)
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {str(e)}")
            return []
            
    async def store_memory(self, text: str) -> str:
        """存储记忆内容"""
        if not text.strip():
            return ""
            
        try:
            # 获取文本的嵌入向量
            embedding = await self.get_embedding(text)
            if not embedding:
                return ""
                
            # 存储段落
            paragraph_hash = self.paragraph_store.add_item(text, embedding)
            
            # 提取三元组
            triples = await self.extract_triples(text)
            
            # 处理三元组
            for subj, rel, obj in triples:
                # 获取实体嵌入向量
                subj_embedding = await self.get_embedding(subj)
                obj_embedding = await self.get_embedding(obj)
                rel_embedding = await self.get_embedding(rel)
                
                # 存储实体和关系
                subj_hash = self.entity_store.add_item(subj, subj_embedding)
                obj_hash = self.entity_store.add_item(obj, obj_embedding)
                rel_hash = self.relation_store.add_item(rel, rel_embedding)
                
                # 更新知识图谱
                if not self.knowledge_graph.has_node(subj_hash):
                    self.knowledge_graph.add_node(subj_hash, type='entity', label=subj)
                    
                if not self.knowledge_graph.has_node(obj_hash):
                    self.knowledge_graph.add_node(obj_hash, type='entity', label=obj)
                    
                # 添加关系边
                self.knowledge_graph.add_edge(subj_hash, obj_hash, type='relation', label=rel, hash=rel_hash)
                
                # 将段落链接到实体
                self.knowledge_graph.add_edge(subj_hash, paragraph_hash, type='contains', weight=1.0)
                self.knowledge_graph.add_edge(obj_hash, paragraph_hash, type='contains', weight=1.0)
            
            # 保存数据
            self._save_data()
            
            return paragraph_hash
        except Exception as e:
            logger.error(f"存储记忆失败: {str(e)}")
            return ""
            
    async def search_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关记忆"""
        if not query.strip():
            return []
            
        try:
            # 获取查询的嵌入向量
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                return []
                
            # 直接搜索段落
            paragraph_results = self.paragraph_store.search(query_embedding, top_k)
            
            # 提取三元组
            query_triples = await self.extract_triples(query)
            
            # 使用三元组增强检索
            enhanced_results = []
            for subj, rel, obj in query_triples:
                # 搜索相关实体
                subj_embedding = await self.get_embedding(subj)
                obj_embedding = await self.get_embedding(obj)
                
                if subj_embedding:
                    subj_results = self.entity_store.search(subj_embedding, top_k=3)
                    for hash_key, score in subj_results:
                        if hash_key in self.knowledge_graph:
                            # 获取与实体相关的段落
                            for _, para_hash in self.knowledge_graph.out_edges(hash_key):
                                if self.knowledge_graph[hash_key][para_hash]['type'] == 'contains':
                                    para_item = self.paragraph_store.get_item(para_hash)
                                    if para_item:
                                        enhanced_results.append((para_hash, score * 0.9))  # 稍微降低权重
                
                if obj_embedding:
                    obj_results = self.entity_store.search(obj_embedding, top_k=3)
                    for hash_key, score in obj_results:
                        if hash_key in self.knowledge_graph:
                            # 获取与实体相关的段落
                            for _, para_hash in self.knowledge_graph.out_edges(hash_key):
                                if self.knowledge_graph[hash_key][para_hash]['type'] == 'contains':
                                    para_item = self.paragraph_store.get_item(para_hash)
                                    if para_item:
                                        enhanced_results.append((para_hash, score * 0.9))  # 稍微降低权重
            
            # 合并直接结果和增强结果
            all_results = paragraph_results + enhanced_results
            
            # 去重并按相似度排序
            result_dict = {}
            for hash_key, score in all_results:
                if hash_key in result_dict:
                    result_dict[hash_key] = max(result_dict[hash_key], score)
                else:
                    result_dict[hash_key] = score
                    
            sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # 构建返回结果
            formatted_results = []
            for hash_key, score in sorted_results:
                item = self.paragraph_store.get_item(hash_key)
                if item:
                    formatted_results.append({
                        "content": item.content,
                        "similarity": score,
                        "hash": hash_key,
                        "source": "lpmm"
                    })
                    
            return formatted_results
        except Exception as e:
            logger.error(f"搜索记忆失败: {str(e)}")
            return []
    
    async def build_memory_from_chat_history(self, days: int = 1, max_samples: int = 100) -> int:
        """从聊天历史构建记忆"""
        try:
            # 计算时间范围
            end_time = time.time()
            start_time = end_time - (days * 24 * 60 * 60)
            
            # 从数据库获取聊天记录
            messages = db.messages.find({
                "time": {"$gte": start_time, "$lte": end_time},
                "memorized_times": {"$lt": 3}  # 限制已记忆次数
            }).sort("time", -1).limit(max_samples)
            
            messages_list = list(messages)
            processed_count = 0
            
            logger.info(f"从聊天历史中找到{len(messages_list)}条消息待处理")
            
            for message in messages_list:
                content = message.get("raw_message", "")
                if not content.strip() or len(content) < 10:
                    continue
                    
                # 存储记忆
                hash_key = await self.store_memory(content)
                if hash_key:
                    processed_count += 1
                    
                    # 更新记忆次数
                    db.messages.update_one(
                        {"_id": message["_id"]},
                        {"$inc": {"memorized_times": 1}}
                    )
            
            logger.info(f"成功处理{processed_count}条消息")
            return processed_count
        except Exception as e:
            logger.error(f"从聊天历史构建记忆失败: {str(e)}")
            return 0
    
    async def remove_old_memories(self, days: int = 30) -> int:
        """移除旧记忆"""
        # 待实现
        return 0


async def create_lpmm_memory_system(memory_graph: Memory_graph, hippocampus: Hippocampus, llm_client: Any) -> LPMMMemoryBuilder:
    """创建LPMM记忆系统
    
    Args:
        memory_graph: 传统记忆图实例
        hippocampus: 海马体实例
        llm_client: LLM客户端
        
    Returns:
        LPMMMemoryBuilder: LPMM记忆构建器实例
    """
    # 创建LPMM记忆构建器
    lpmm_builder = LPMMMemoryBuilder(
        data_dir=os.path.join(os.path.dirname(__file__), "data/lpmm"),
        llm_client=llm_client
    )
    
    # 返回构建器
    return lpmm_builder
    
async def query_lpmm_memories(lpmm_builder: LPMMMemoryBuilder, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """查询LPMM记忆
    
    Args:
        lpmm_builder: LPMM记忆构建器实例
        query: 查询文本
        top_k: 返回的最大记忆数量
        
    Returns:
        List[Dict[str, Any]]: 记忆列表，每个记忆包含内容、相似度等信息
    """
    if not lpmm_builder:
        logger.warning("LPMM记忆构建器未初始化")
        return []
        
    # 查询记忆
    memories = await lpmm_builder.search_memory(query, top_k=top_k)
    return memories 


# ================== 调试功能 ==================
async def debug_main():
    """调试入口点，用于在开发阶段测试LPMM记忆系统的功能"""
    logger.info("===== LPMM记忆系统调试模式 =====")
    
    # 创建LLM客户端
    try:
        from src.plugins.models.utils_model import LLM_request
        
        llm_client = LLM_request(
            model=global_config.llm_model,
            temperature=0.2,
            request_type="chat"
        )
        logger.success("成功创建LLM客户端")
    except Exception as e:
        logger.error(f"创建LLM客户端失败: {e}")
        # 创建一个模拟的LLM客户端
        class MockLLMClient:
            async def generate_response(self, prompt: str):
                logger.info(f"模拟LLM请求：{prompt[:50]}...")
                triples_data = [
                    ["用户", "询问", "天气"],
                    ["今天", "是", "晴天"]
                ]
                return json.dumps(triples_data), "", ""
            
            async def send_embedding_request(self, text: str):
                logger.info(f"模拟Embedding请求：{text[:50]}...")
                # 返回一个随机的embedding向量(768维)
                return [random.random() * 0.1 for _ in range(768)]
        
        llm_client = MockLLMClient()
        logger.warning("使用模拟的LLM客户端")
    
    # 创建LPMM记忆构建器
    try:
        lpmm_builder = LPMMMemoryBuilder(
            data_dir=os.path.join(os.path.dirname(__file__), "data/debug_lpmm"),
            llm_client=llm_client
        )
        logger.success("成功创建LPMM记忆构建器")
    except Exception as e:
        logger.error(f"创建LPMM记忆构建器失败: {e}")
        return
    
    while True:
        print("\n===== LPMM调试菜单 =====")
        print("1. 添加记忆")
        print("2. 查询记忆")
        print("3. 建立记忆（从聊天历史）")
        print("4. 记忆维护")
        print("5. 退出")
        choice = input("请选择操作 (1-5): ").strip()
        
        if choice == "1":
            text = input("请输入要添加的记忆内容: ").strip()
            if text:
                hash_key = await lpmm_builder.store_memory(text)
                if hash_key:
                    logger.success(f"成功添加记忆，哈希值: {hash_key}")
                else:
                    logger.error("添加记忆失败")
        
        elif choice == "2":
            query = input("请输入查询内容: ").strip()
            if query:
                memories = await lpmm_builder.search_memory(query)
                if memories:
                    print("\n=== 检索结果 ===")
                    for i, memory in enumerate(memories, 1):
                        print(f"\n【结果 {i}】相似度: {memory['similarity']:.4f}")
                        print(memory["content"])
                else:
                    logger.warning("未找到相关记忆")
        
        elif choice == "3":
            days = int(input("处理多少天的聊天记录 (默认1): ") or "1")
            max_samples = int(input("处理的最大样本数 (默认50): ") or "50")
            
            processed_count = await lpmm_builder.build_memory_from_chat_history(
                days=days, 
                max_samples=max_samples
            )
            logger.info(f"处理了 {processed_count} 条记忆")
            
        elif choice == "4":
            days = int(input("清理多少天前的记忆 (默认30): ") or "30")
            removed_count = await lpmm_builder.remove_old_memories(days=days)
            logger.info(f"移除了 {removed_count} 条旧记忆")
            
        elif choice == "5":
            logger.info("退出调试模式")
            break
            
        else:
            logger.warning("无效的选择，请重新输入")


if __name__ == "__main__":
    # 直接运行文件时执行调试功能
    asyncio.run(debug_main()) 