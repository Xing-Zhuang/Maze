import os
import ray
import argparse
from typing import Dict, List
from collections import Counter
from maze.agent.config import config
import logging

@ray.remote
class DAGContext:
    """
    DAGContext class to store and manage DAG-related data.
    """

    def __init__(self, dag_id: str, node_id: str, node_ip: str)-> None:
        """
        Initialize DAGContext with DAG ID, preferred node ID, and preferred node IP.

        :param dag_id: Unique identifier for the DAG
        :param node_id: Preferred node ID for task execution
        :param node_ip: Preferred node IP address for task execution
        """
        self.data= {}
        self.data["dag_id"]= ray.put(dag_id)
        self.dag_id= dag_id
        self.preferred_node_ip= node_ip
        self.preferred_node_id= node_id

    def put(self, key: str, value: object)-> None:
        """
        Store a value in the DAG context using a key.

        :param key: The key for the data
        :param value: The value to store in the context
        """
        self.data[key]= ray.put(value)

    def get(self, key: str)-> object:
        """
        Retrieve a value from the DAG context using a key.

        :param key: The key for the data
        :return: The value associated with the key
        :raises KeyError: If the key is not found in the context
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found.")
        return ray.get(self.data[key])

    def get_preferred_id(self)-> str:
        """
        Get the preferred node ID for task execution.

        :return: Preferred node ID
        """
        return self.preferred_node_id

class DAGContextManager:
    """
    不再依赖args对象，直接从全局config单例获取配置。
    """
    def __init__(self):
        self.run2ctx: Dict[str, ray.actor.ActorHandle] = {}
        self.ctx2id: Dict[ray.actor.ActorHandle, str] = {}
        self.node_load_counter: Dict[str, int] = Counter()
        self.logger = logging.getLogger(__name__)

    def get_context(self, run_id: str) -> ray.actor.ActorHandle:
        return self.run2ctx.get(run_id, None)

    def create_context(self, node_id: str, node_ip: str, task_info: dict) -> ray.actor.ActorHandle:
        """
        创建并返回一个新的 DAGContext actor 实例，并从全局 config 预加载所有配置。
        """
        run_id = task_info.get('run_id')
        dag_id = task_info.get('dag_id')
        dag_ctx = DAGContext.remote(dag_id, node_id, node_ip)
        self.run2ctx[run_id] = dag_ctx
        self.ctx2id[dag_ctx] = node_id
        self.node_load_counter[node_id] += 1
        self.logger.debug(f"  -> Context created on node '{node_id}'. Current node loads: {dict(self.node_load_counter)}")
        self.logger.debug("  -> Injecting configurations into DAGContext...")
        put_futures = []
        # 1. 注入 [paths] 配置
        paths_config = config.get('paths', {})
        project_root = paths_config.get('project_root')
        model_folder = paths_config.get('model_cache_dir')
        if project_root and model_folder:
            put_futures.append(dag_ctx.put.remote("model_cache_dir", os.path.join(project_root, "maze", model_folder)))
        # 2. 注入 [server] 配置
        server_config = config.get('server', {})
        for key, value in server_config.items():
            put_futures.append(dag_ctx.put.remote(key, value))
        # 3. 注入 [online_apis] 配置
        for api_profile in config.online_apis.values():
            for key, value in api_profile.items():
                put_futures.append(dag_ctx.put.remote(key, value))
        # 4. 注入 task_info 中的运行时信息
        put_futures.append(dag_ctx.put.remote("arrival_time", task_info.get('arrival_time')))
        # 等待所有注入操作完成
        ray.get(put_futures)
        self.logger.debug("  -> ✅ Configurations successfully injected.")
        return dag_ctx

    def release_context(self, run_id: str) -> bool:
        """释放并移除与 run_id 关联的 DAGContext actor。"""
        ctx = self.run2ctx.pop(run_id, None)
        if ctx:
            node_id = self.ctx2id.pop(ctx, None)
            if node_id and node_id in self.node_load_counter:
                self.node_load_counter[node_id] -= 1
                if self.node_load_counter[node_id] == 0:
                    del self.node_load_counter[node_id]
                self.logger.debug(f"  -> Context released from node '{node_id}'. Current node loads: {dict(self.node_load_counter)}")
            ray.kill(ctx)
            return True
        return False

    def get_least_loaded_node(self, available_nodes: List[str]):
        """从可用节点列表中找到负载最小的节点。"""
        if not available_nodes:
            return None
        return min(available_nodes, key=lambda node_id: self.node_load_counter.get(node_id, 0))

    def get_all_contexts(self) -> Dict[str, ray.actor.ActorHandle]:
        """获取所有当前活动的 DAGContexts。"""
        return self.run2ctx