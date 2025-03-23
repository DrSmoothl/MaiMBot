import asyncio
import datetime
import time
from typing import List, Dict, Any
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.common.logger import get_module_logger
from src.plugins.chat.config import global_config
from src.plugins.memory_system.memory import Memory_graph, Hippocampus, LLM_request
from src.plugins.memory_system.lpmm_memory_builder import LPMMMemoryBuilder, query_lpmm_memories, create_lpmm_memory_system

logger = get_module_logger("lpmm_memory_integration")

# 延迟初始化driver和config
_driver = None
_config = None

def get_config():
    global _driver, _config
    if _driver is None:
        from nonebot import get_driver
        _driver = get_driver()
        _config = _driver.config
    return _config

class MemoryClient:
    """记忆系统客户端，集成传统记忆系统和LPMM记忆系统"""
    
    def __init__(self):
        # 初始化传统记忆图
        self.memory_graph = Memory_graph()
        self.hippocampus = Hippocampus(self.memory_graph)
        
        # 初始化LLM客户端
        self.llm_client = LLM_request(
            model=global_config.llm_model,
            temperature=0.2,
            request_type="chat"
        )
        
        # LPMM记忆构建器
        self.lpmm_builder = None
        self.is_initialized = False
        
    async def initialize(self):
        """初始化记忆系统"""
        if self.is_initialized:
            return
            
        # 从数据库同步现有记忆
        logger.info("从数据库同步记忆...")
        self.hippocampus.sync_memory_from_db()
        
        # 如果LPMM记忆系统已禁用，则不创建
        if not global_config.lpmm_enabled:
            logger.info("LPMM记忆系统已禁用，仅使用传统记忆系统")
            self.is_initialized = True
            return
            
        # 创建LPMM记忆系统
        logger.info("创建LPMM记忆系统...")
        try:
            self.lpmm_builder = await create_lpmm_memory_system(
                self.memory_graph,
                self.hippocampus,
                self.llm_client
            )
            logger.success("LPMM记忆系统创建成功")
        except Exception as e:
            logger.error(f"创建LPMM记忆系统失败: {e}")
            # 如果LPMM初始化失败，仍然可以使用传统记忆系统
            
        self.is_initialized = True
        
    async def build_memory(self):
        """构建记忆"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # 首先使用传统记忆构建方法
            await self.hippocampus.operation_build_memory()
            
            # 然后如果LPMM构建器存在，也执行LPMM的记忆构建
            if self.lpmm_builder:
                await self.lpmm_builder.build_memory_from_chat_history()
                
            logger.success("记忆构建完成")
            return True
        except Exception as e:
            logger.error(f"记忆构建失败: {e}")
            return False
            
    async def query_memory(self, query: str, max_memories: int = 5) -> List[Dict[str, Any]]:
        """查询记忆"""
        if not self.is_initialized:
            await self.initialize()
            
        # 结果列表
        results = []
        
        # 首先尝试使用LPMM检索（如果可用）
        if self.lpmm_builder:
            try:
                lpmm_memories = await query_lpmm_memories(self.lpmm_builder, query, top_k=max_memories)
                for memory in lpmm_memories:
                    # 确保记忆源标记正确
                    if 'source' not in memory:
                        memory['source'] = 'lpmm'
                results.extend(lpmm_memories)
                logger.info(f"LPMM检索到 {len(lpmm_memories)} 条记忆")
            except Exception as e:
                logger.error(f"LPMM记忆检索失败: {e}")
        
        # 然后使用传统记忆系统检索
        try:
            traditional_memories = await self.hippocampus.get_relevant_memories(
                text=query, 
                max_topics=5, 
                similarity_threshold=0.4,
                max_memory_num=max_memories
            )
            
            # 转换格式，确保与LPMM结果格式一致
            for memory in traditional_memories:
                # 添加时间戳（如果有）
                timestamp = None
                if 'created_time' in memory:
                    timestamp = memory['created_time']
                elif 'last_modified' in memory:
                    timestamp = memory['last_modified']
                
                memory_item = {
                    "content": memory["content"],
                    "similarity": memory["similarity"],
                    "topic": memory.get("topic", "未知主题"),
                    "source": "traditional"
                }
                
                if timestamp:
                    memory_item["timestamp"] = timestamp
                    
                results.append(memory_item)
                
            logger.info(f"传统系统检索到 {len(traditional_memories)} 条记忆")
        except Exception as e:
            logger.error(f"传统记忆检索失败: {e}")
            
        # 去重并按相似度排序
        unique_contents = set()
        filtered_results = []
        
        for memory in results:
            content_hash = self._get_content_hash(memory["content"])
            if content_hash not in unique_contents:
                unique_contents.add(content_hash)
                filtered_results.append(memory)
                
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 限制返回数量
        return filtered_results[:max_memories]
    
    def _get_content_hash(self, content: str) -> str:
        """获取内容的哈希值，用于去重"""
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()
        
    async def _lpmm_maintenance(self):
        """LPMM记忆维护任务"""
        if not self.lpmm_builder:
            return
            
        try:
            # 从最近的聊天历史构建新记忆
            days = global_config.memory_update_days if hasattr(global_config, 'memory_update_days') else 1
            max_samples = global_config.memory_update_samples if hasattr(global_config, 'memory_update_samples') else 100
            
            processed_count = await self.lpmm_builder.build_memory_from_chat_history(
                days=days, 
                max_samples=max_samples
            )
            
            logger.info(f"LPMM记忆维护：处理了{processed_count}条新消息")
            
            # 移除旧记忆（可选）
            if hasattr(global_config, 'memory_retention_days') and global_config.memory_retention_days > 0:
                removed_count = await self.lpmm_builder.remove_old_memories(
                    days=global_config.memory_retention_days
                )
                if removed_count > 0:
                    logger.info(f"LPMM记忆维护：移除了{removed_count}条旧记忆")
        except Exception as e:
            logger.error(f"LPMM记忆维护失败: {e}")
        
    async def memory_maintenance(self):
        """执行记忆维护任务：合并、遗忘等"""
        if not self.is_initialized:
            await self.initialize()
            
        # 执行传统记忆维护
        await self.hippocampus.operation_merge_memory(percentage=0.1)
        await self.hippocampus.operation_forget_topic(percentage=0.05)
        
        # 执行LPMM记忆维护
        await self._lpmm_maintenance()
        
        logger.success("记忆维护完成")

# 使用示例
async def main():
    # 创建记忆客户端
    memory_client = MemoryClient()
    
    # 初始化
    await memory_client.initialize()
    
    # 构建记忆
    await memory_client.build_memory()
    
    # 查询记忆示例
    memories = await memory_client.query_memory("你昨天和我聊了什么？")
    
    # 打印结果
    print("\n检索到的相关记忆:")
    for i, memory in enumerate(memories, 1):
        print(f"\n--- 记忆 {i} (相似度: {memory['similarity']:.4f}) ---")
        print(memory["content"])
        
    # 执行记忆维护
    await memory_client.memory_maintenance()
    
    
if __name__ == "__main__":
    asyncio.run(main()) 