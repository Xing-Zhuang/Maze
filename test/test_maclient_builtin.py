"""
测试使用内置任务的简单工作流
使用 pytest 运行: pytest test/test_simple_api.py -v
"""

import pytest
from maze import MaClient
from maze.client.maze.builtin import simpleTask


@pytest.fixture
def client():
    """创建并返回 MaClient 实例"""
    return MaClient()


@pytest.fixture
def workflow(client):
    """创建并返回工作流实例"""
    return client.create_workflow()


class TestSimpleWorkflow:
    """测试简单工作流的执行"""
    
    def test_builtin_task_workflow(self, workflow):
        """测试使用内置任务的工作流执行"""
        # 添加任务1
        task1 = workflow.add_task(
            simpleTask.task1,
            inputs={"task1_input": "这是task1的输入"}
        )
        
        # 添加任务2，引用task1的输出
        task2 = workflow.add_task(
            simpleTask.task2,
            inputs={"task2_input": task1.outputs["task1_output"]}
        )
        
        # 添加任务依赖关系
        workflow.add_edge(task1, task2)
        
        # 运行工作流并获取run_id
        run_id = workflow.run()
        print(f"Workflow started with run_id: {run_id}")
        
        # 获取并显示执行结果（格式化输出）
        results = workflow.show_results(run_id)
        task_results = results["task_results"]
        workflow_completed = results["workflow_completed"]
        
        # 断言：工作流应该完成
        assert workflow_completed, "工作流未完成"
        
        # 断言：应该有2个任务的结果
        assert len(task_results) == 2, f"期望2个任务结果，实际得到 {len(task_results)} 个"
        
        # 断言：每个任务都应该有输出
        for task_id, result in task_results.items():
            assert result is not None, f"任务 {task_id} 的结果为空"
            assert isinstance(result, dict), f"任务 {task_id} 的结果应该是字典类型"
        
        print(f"\n✓ 测试通过: 工作流成功执行，完成 {len(task_results)} 个任务")
    
    def test_task_output_propagation(self, workflow):
        """测试任务输出是否正确传递到下游任务"""
        # 添加任务1
        task1 = workflow.add_task(
            simpleTask.task1,
            inputs={"task1_input": "测试输入-"}
        )
        
        # 添加任务2
        task2 = workflow.add_task(
            simpleTask.task2,
            inputs={"task2_input": task1.outputs["task1_output"]}
        )
        
        # 添加依赖关系
        workflow.add_edge(task1, task2)
        
        # 运行工作流并获取run_id
        run_id = workflow.run()
        print(f"Workflow started with run_id: {run_id}")
        
        # 获取并显示执行结果
        results = workflow.show_results(run_id)
        task_results = results["task_results"]
        
        # 断言：两个任务都应该有结果
        assert len(task_results) == 2
        
        # 验证task1的输出包含输入字符串
        task1_result = list(task_results.values())[0]
        assert "task1_output" in task1_result
        assert "测试输入-" in task1_result["task1_output"]
        
        # 验证task2的输出包含task1的输出内容
        task2_result = list(task_results.values())[1]
        assert "task2_output" in task2_result
        assert "====" in task2_result["task2_output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


