/**
 * API服务层 - 适配新的后端接口
 * 基于 server.py 中的API端点
 */

class ApiService {
  constructor() {
    this.baseUrl = 'http://localhost:8000';
  }

  /**
   * 创建工作流
   * POST /create_workflow
   */
  async createWorkflow() {
    try {
      const response = await axios.post(`${this.baseUrl}/create_workflow`);
      return response.data;
    } catch (error) {
      console.error('创建工作流API错误:', error);
      throw new Error(`创建工作流失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 添加任务到工作流
   * POST /add_task
   * @param {string} workflowId - 工作流ID
   * @param {string} taskType - 任务类型 (目前只支持 'code')
   */
  async addTask(workflowId, taskType = 'code') {
    try {
      const response = await axios.post(`${this.baseUrl}/add_task`, {
        workflow_id: workflowId,
        task_type: taskType
      });
      return response.data;
    } catch (error) {
      console.error('添加任务API错误:', error);
      throw new Error(`添加任务失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 删除任务
   * POST /del_task
   * @param {string} workflowId - 工作流ID
   * @param {string} taskId - 任务ID
   */
  async deleteTask(workflowId, taskId) {
    try {
      const response = await axios.post(`${this.baseUrl}/del_task`, {
        workflow_id: workflowId,
        task_id: taskId
      });
      return response.data;
    } catch (error) {
      throw new Error(`删除任务失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 保存任务详细信息
   * POST /save_task
   * @param {string} workflowId - 工作流ID
   * @param {string} taskId - 任务ID
   * @param {Object} taskInput - 任务输入参数
   * @param {Object} taskOutput - 任务输出参数
   * @param {Object} resources - 所需资源
   * @param {string} codeStr - 任务代码
   */
  async saveTask(workflowId, taskId, taskInput, taskOutput, resources, codeStr) {
    try {
      const response = await axios.post(`${this.baseUrl}/save_task`, {
        workflow_id: workflowId,
        task_id: taskId,
        task_input: taskInput,
        task_output: taskOutput,
        resources: resources,
        code_str: codeStr
      });
      return response.data;
    } catch (error) {
      throw new Error(`保存任务失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 添加边（任务依赖关系）
   * POST /add_edge
   * @param {string} workflowId - 工作流ID
   * @param {string} sourceTaskId - 源任务ID
   * @param {string} targetTaskId - 目标任务ID
   */
  async addEdge(workflowId, sourceTaskId, targetTaskId) {
    try {
      const response = await axios.post(`${this.baseUrl}/add_edge`, {
        workflow_id: workflowId,
        source_task_id: sourceTaskId,
        target_task_id: targetTaskId
      });
      return response.data;
    } catch (error) {
      throw new Error(`添加边失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 删除边
   * POST /del_edge
   * @param {string} workflowId - 工作流ID
   * @param {string} sourceTaskId - 源任务ID
   * @param {string} targetTaskId - 目标任务ID
   */
  async deleteEdge(workflowId, sourceTaskId, targetTaskId) {
    try {
      const response = await axios.post(`${this.baseUrl}/del_edge`, {
        workflow_id: workflowId,
        source_task_id: sourceTaskId,
        target_task_id: targetTaskId
      });
      return response.data;
    } catch (error) {
      throw new Error(`删除边失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 运行工作流
   * POST /run_workflow
   * @param {string} workflowId - 工作流ID
   */
  async runWorkflow(workflowId) {
    try {
      const response = await axios.post(`${this.baseUrl}/run_workflow`, {
        workflow_id: workflowId
      });
      return response.data;
    } catch (error) {
      throw new Error(`运行工作流失败: ${error.response?.data?.message || error.message}`);
    }
  }

  /**
   * 获取工作流运行结果
   * WebSocket /get_workflow_res/{workflow_id}
   * @param {string} workflowId - 工作流ID
   * @param {Function} onMessage - 消息回调函数
   * @param {Function} onError - 错误回调函数
   * @param {Function} onClose - 关闭回调函数
   */
  getWorkflowResults(workflowId, onMessage, onError, onClose) {
    const wsUrl = `ws://localhost:8000/get_workflow_res/${workflowId}`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log(`WebSocket连接已建立: ${wsUrl}`);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('解析WebSocket消息失败:', error);
        onError(error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket错误:', error);
      onError(error);
    };
    
    ws.onclose = (event) => {
      console.log('WebSocket连接已关闭:', event.code, event.reason);
      onClose(event);
    };
    
    return ws;
  }
}

// 导出API服务实例
window.ApiService = ApiService;
