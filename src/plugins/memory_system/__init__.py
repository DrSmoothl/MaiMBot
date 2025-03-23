"""记忆系统模块，包含基于图的记忆系统和LPMM记忆系统的集成实现"""

import asyncio
import sys
from src.common.logger import get_module_logger

# 配置日志
logger = get_module_logger("memory_system_init")

# 创建传统记忆系统实例 - 延迟导入以避免循环依赖
memory_graph = None
hippocampus = None

# 创建集成记忆系统客户端
memory_client = None
_memory_system_initialized = False

# 全局变量保存调度器引用
scheduler = None

# 延迟导入，避免循环依赖
def init_scheduler():
    """初始化调度器"""
    global scheduler
    try:
        from nonebot import require
        scheduler = require("nonebot_plugin_apscheduler").scheduler
        return scheduler
    except (ImportError, RuntimeError) as e:
        logger.warning(f"无法导入定时任务插件: {e}")
        return None

def init_memory_instances():
    """初始化记忆系统实例，延迟导入以避免循环依赖"""
    global memory_graph, hippocampus, memory_client, _memory_system_initialized
    
    if memory_graph is None:
        try:
            from .memory import Memory_graph, Hippocampus
            
            # 此处需要导入config但有循环导入风险，延迟导入
            try:
                from src.plugins.chat.config import global_config
            except ImportError:
                logger.warning("无法导入config，使用默认值")
                # 创建一个空对象作为默认config
                class DummyConfig:
                    lpmm_enabled = False
                global_config = DummyConfig()
            
            memory_graph = Memory_graph()
            hippocampus = Hippocampus(memory_graph)
            
            # 仅当LPMM启用时才导入相关模块
            if hasattr(global_config, 'lpmm_enabled') and global_config.lpmm_enabled:
                try:
                    from .lpmm_memory_integration import MemoryClient
                    memory_client = MemoryClient()
                except ImportError as e:
                    logger.warning(f"无法导入LPMM记忆系统: {e}")
        except ImportError as e:
            logger.error(f"初始化记忆系统失败: {e}")
            return None, None, None
        
    return memory_graph, hippocampus, memory_client

async def ensure_memory_system_initialized():
    """确保记忆系统已初始化"""
    global _memory_system_initialized, memory_client
    
    if memory_client is None:
        init_memory_instances()
    
    if not _memory_system_initialized and memory_client is not None:
        try:
            logger.info("初始化集成记忆系统...")
            await memory_client.initialize()
            logger.success("集成记忆系统创建成功")
            _memory_system_initialized = True
        except Exception as e:
            logger.error(f"初始化集成记忆系统失败: {e}")
            # 如果集成系统初始化失败，仍然可以使用传统记忆系统

# 延迟设置定时任务
def setup_tasks():
    """设置所有定时任务"""
    global scheduler
    
    # 获取调度器
    if scheduler is None:
        scheduler = init_scheduler()
        
    if scheduler is None:
        logger.warning("无法初始化定时任务")
        return
            
    # 初始化记忆系统实例
    init_memory_instances()
    
    # 获取全局配置
    try:
        from src.plugins.chat.config import global_config
    except ImportError:
        logger.warning("无法导入config，使用默认值")
        # 创建一个空对象作为默认config
        class DummyConfig:
            lpmm_enabled = False
            lpmm_build_interval = 1800
            lpmm_maintain_interval = 7200
        global_config = DummyConfig()
    
    # 启动时初始化记忆系统
    @scheduler.scheduled_job("interval", seconds=60, id="init_memory_system")
    async def init_memory_system():
        """定时初始化记忆系统的任务"""
        global _memory_system_initialized
        
        if not _memory_system_initialized:
            await ensure_memory_system_initialized()
            # 成功初始化后移除此任务
            if _memory_system_initialized:
                scheduler.remove_job("init_memory_system")
    
    # 只有在lpmm_enabled为True时才添加LPMM相关任务
    if hasattr(global_config, 'lpmm_enabled') and global_config.lpmm_enabled:
        # 设置LPMM相关的定时任务
        try:
            # 定期构建LPMM记忆的任务
            @scheduler.scheduled_job("interval", seconds=global_config.lpmm_build_interval, id="build_lpmm_memory")
            async def build_lpmm_memory_task():
                """每lpmm_build_interval秒执行一次LPMM记忆构建"""
                global _memory_system_initialized
                
                if not _memory_system_initialized:
                    await ensure_memory_system_initialized()
                    
                if _memory_system_initialized and memory_client.lpmm_builder:
                    logger.debug("[LPMM记忆构建]------------------------------------开始构建LPMM记忆--------------------------------------")
                    try:
                        start_time = asyncio.get_event_loop().time()
                        await memory_client.build_memory()
                        end_time = asyncio.get_event_loop().time()
                        logger.success(
                            f"[LPMM记忆构建]--------------------------LPMM记忆构建完成：耗时: {end_time - start_time:.2f} "
                            "秒-------------------------------------------"
                        )
                    except Exception as e:
                        logger.error(f"构建LPMM记忆失败: {e}")

            # 定期维护LPMM记忆的任务
            @scheduler.scheduled_job("interval", seconds=global_config.lpmm_maintain_interval, id="maintain_lpmm_memory")
            async def maintain_lpmm_memory_task():
                """每lpmm_maintain_interval秒执行一次LPMM记忆维护"""
                global _memory_system_initialized
                
                if not _memory_system_initialized:
                    await ensure_memory_system_initialized()
                    
                if _memory_system_initialized and memory_client.lpmm_builder:
                    logger.debug("[LPMM记忆维护]------------------------------------开始维护LPMM记忆--------------------------------------")
                    try:
                        start_time = asyncio.get_event_loop().time()
                        await memory_client.memory_maintenance()
                        end_time = asyncio.get_event_loop().time()
                        logger.success(
                            f"[LPMM记忆维护]--------------------------LPMM记忆维护完成：耗时: {end_time - start_time:.2f} "
                            "秒-------------------------------------------"
                        )
                    except Exception as e:
                        logger.error(f"维护LPMM记忆失败: {e}")
            
            logger.success("LPMM记忆系统任务已设置")
        except Exception as e:
            logger.error(f"设置LPMM任务失败: {e}")
    else:
        logger.info("LPMM记忆系统已禁用，仅使用传统记忆系统")

# 延迟设置任务，这会在NoneBot初始化完成后由dispatcher调用
def setup_plugin():
    """在NoneBot初始化后被调用来设置插件"""
    try:
        # 初始化记忆实例
        init_memory_instances()
        # 设置定时任务
        setup_tasks()
    except Exception as e:
        logger.error(f"插件设置失败: {e}")

# 为了在导入时不抛出异常，使用try-except包裹初始化代码
try:
    # 初始化记忆实例，如果出错会在函数内处理
    init_memory_instances()
except Exception as e:
    # 这个异常已经在函数内部记录，这里不需要再处理
    pass

# 当NoneBot加载插件后，这个函数会被调用
from nonebot import get_driver
driver = get_driver()

@driver.on_startup
async def _():
    """在NoneBot启动时设置插件"""
    setup_plugin()
