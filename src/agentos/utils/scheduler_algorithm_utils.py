import networkx as nx
from typing import Dict, Tuple
# 当任务的历史执行时间未知时，使用的默认执行时间（单位：秒）
TASK_TYPE_DEFAULT_EXEC_TIMES = {'io': 2.0, 'cpu': 5.0, 'gpu': 30.0, 'default': 1.0}

def compute_ranku(dag: nx.DiGraph, exec_time_db: Dict[Tuple[str, str], float], dag_id: str) -> Dict[str, float]:
    """
    使用DFS(深度优先搜索)自底向上计算DAG中每个任务的ranku值（向上排名）。
    ranku(i) = w(i) + max_{j in succ(i)}(ranku(j))
    其中w(i)是任务i的执行时间，succ(i)是任务i的直接后继任务集合。
    出口任务的ranku值就是其自身的执行时间。
    Args:
        dag (nx.DiGraph): 代表工作流的NetworkX有向无环图。
        exec_time_db (Dict): 存储任务历史平均执行时间的字典。
                                格式为 {(dag_id, task_name): exec_time}。
        dag_id (str): 当前DAG的模板ID。

    Returns:
        Dict[str, float]: 一个字典，映射每个任务名到其计算出的ranku值。
    """
    ranku_cache = {}
    def dfs(task_name: str) -> float:
        # 如果已经计算过，直接返回缓存结果（记忆化搜索）
        if task_name in ranku_cache:
            return ranku_cache[task_name]
        # 从数据库获取任务执行时间，若无则使用默认值
        exec_time = exec_time_db.get((dag_id, task_name))
        if exec_time is None:
            task_type = dag.nodes[task_name].get('type')
            exec_time = TASK_TYPE_DEFAULT_EXEC_TIMES.get(task_type, TASK_TYPE_DEFAULT_EXEC_TIMES['default'])

        # 获取所有直接后继任务
        successors = list(dag.successors(task_name))
        # 如果没有后继（出口任务），则ranku值就是其自身执行时间
        if not successors:
            ranku_cache[task_name] = exec_time
            return exec_time
        # 否则，ranku值等于自身执行时间 + 最大的后继ranku值
        max_succ_ranku = max(dfs(s_task) for s_task in successors)
        ranku_cache[task_name] = exec_time + max_succ_ranku
        return ranku_cache[task_name]
    # 遍历图中所有节点，确保每个连通分量的ranku都被计算
    for node in dag.nodes:
        if node not in ranku_cache:
            dfs(node)
    return ranku_cache


def compute_rankd(dag: nx.DiGraph, exec_time_db: Dict[Tuple[str, str], float], dag_id: str) -> Dict[str, float]:
    """
    使用DFS(深度优先搜索)自顶向下计算DAG中每个任务的rankd值（向下排名）。
    rankd(i) = max_{j in pred(i)}(rankd(j) + w(j))
    其中w(j)是任务j的执行时间，pred(i)是任务i的直接前驱任务集合。
    入口任务的rankd值为0。

    Args:
        dag (nx.DiGraph): 代表工作流的NetworkX有向无环图。
        exec_time_db (Dict): 存储任务历史平均执行时间的字典。
                                格式为 {(dag_id, task_name): exec_time}。
        dag_id (str): 当前DAG的模板ID。

    Returns:
        Dict[str, float]: 一个字典，映射每个任务名到其计算出的rankd值。
    """
    rankd_cache = {}
    def dfs(task_name: str) -> float:
        # 如果已经计算过，直接返回缓存结果
        if task_name in rankd_cache:
            return rankd_cache[task_name]
        # 获取所有直接前驱任务
        predecessors = list(dag.predecessors(task_name))
        # 如果没有前驱（入口任务），则rankd值为0
        if not predecessors:
            rankd_cache[task_name] = 0
            return 0
        exec_time = exec_time_db.get((dag_id, task_name))
        if exec_time is None:
            task_type = dag.nodes[task_name].get('type')
            exec_time = TASK_TYPE_DEFAULT_EXEC_TIMES.get(task_type, TASK_TYPE_DEFAULT_EXEC_TIMES['default'])
        # 否则，rankd值等于 "rankd(j) + w(j)" 的最大值
        max_pred_rankd = max(
            dfs(p_task)+ exec_time
            for p_task in predecessors
        )
        rankd_cache[task_name] = max_pred_rankd
        return rankd_cache[task_name]
    # 遍历图中所有节点，确保每个连通分量的rankd都被计算
    for node in dag.nodes:
        if node not in rankd_cache:
            dfs(node)
    return rankd_cache