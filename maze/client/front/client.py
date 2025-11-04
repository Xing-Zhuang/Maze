import requests
from typing import Optional
from maze.client.front.workflow import MaWorkflow


class MaClient:
    """
    Maze客户端，用于连接Maze服务器并管理工作流
    
    示例:
        # 本地工作流
        client = MaClient("http://localhost:8000")
        workflow = client.create_workflow()
        
        # Agent服务
        client = MaClient("http://localhost:8000", agent_port=8001)
        workflow = client.create_workflow(name="my_agent", mode="server")
    """
    
    def __init__(self, server_url: str = "http://localhost:8000", agent_port: int = 8001):
        """
        初始化Maze客户端
        
        Args:
            server_url: Maze服务器地址，默认为 http://localhost:8000
            agent_port: Agent服务端口，默认为 8001（仅在ServerWorkflow模式下使用）
        """
        self.server_url = server_url.rstrip('/')
        self.agent_port = agent_port
        
    def create_workflow(self) -> MaWorkflow:
        """
        创建本地工作流（LocalWorkflow）
        
        运行一次即完成的工作流
        
        Returns:
            MaWorkflow: 工作流对象
            
        Raises:
            Exception: 如果创建失败
        """
        url = f"{self.server_url}/create_workflow"
        response = requests.post(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                workflow_id = data["workflow_id"]
                return MaWorkflow(workflow_id, self.server_url)
            else:
                raise Exception(f"创建工作流失败: {data.get('message', 'Unknown error')}")
        else:
            raise Exception(f"请求失败，状态码：{response.status_code}, 响应：{response.text}")
    
    def create_server_workflow(self, name: str) -> 'ServerWorkflow':
        """
        创建服务工作流（ServerWorkflow）
        
        可作为Agent持续运行的工作流
        
        Args:
            name: 工作流名称（用于API路径）
            
        Returns:
            ServerWorkflow: 服务工作流对象
            
        示例:
            workflow = client.create_server_workflow(name="my_agent")
            task = workflow.add_task(func, inputs={"user_input": None})
            workflow.deploy()
        """
        from maze.client.server_workflow import ServerWorkflow
        return ServerWorkflow(name, self.server_url, self.agent_port)
    
    def get_workflow(self, workflow_id: str) -> MaWorkflow:
        """
        获取已存在的工作流对象
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            MaWorkflow: 工作流对象
        """
        return MaWorkflow(workflow_id, self.server_url)
    
    def get_ray_head_port(self) -> dict:
        """
        获取Ray头节点端口（用于worker连接）
        
        Returns:
            dict: 包含端口信息的字典
        """
        url = f"{self.server_url}/get_head_ray_port"
        response = requests.post(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取Ray端口失败，状态码：{response.status_code}")

