class ToolbarComponent {
  constructor(container) {
    this.container = container;
    this.apiService = window.api;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.innerHTML = `
      <div class="toolbar">
        <div class="toolbar-left">
          <div class="logo">
            <i class="fas fa-project-diagram"></i>
            <span>Maze Workflow Playground</span>
          </div>
        </div>
        <div class="toolbar-right">
          <button class="btn btn-primary" id="create-workflow-btn">
            <i class="fas fa-plus"></i> 新建工作流
          </button>
          <button class="btn btn-info" id="add-task-btn" disabled>
            <i class="fas fa-plus-circle"></i> 添加任务
          </button>
          <button class="btn btn-success" id="submit-workflow-btn" disabled>
            <i class="fas fa-play"></i> 运行
          </button>
          <button class="btn btn-secondary" id="save-workflow-btn" disabled>
            <i class="fas fa-save"></i> 保存
          </button>
        </div>
      </div>
    `;
  }

  bindEvents() {
    // 创建工作流
    this.container.querySelector('#create-workflow-btn').addEventListener('click', () => {
      this.handleCreateWorkflow();
    });

    // 添加任务
    this.container.querySelector('#add-task-btn').addEventListener('click', () => {
      this.handleAddTask();
    });

    // 运行工作流
    this.container.querySelector('#submit-workflow-btn').addEventListener('click', () => {
      this.handleRunWorkflow();
    });

    // 保存工作流
    this.container.querySelector('#save-workflow-btn').addEventListener('click', () => {
      this.handleSaveWorkflow();
    });

    // 监听状态变化
    window.stateManager.subscribe((newState) => {
      this.updateUI(newState);
    });
  }

  async handleCreateWorkflow() {
    try {
      const result = await this.apiService.createWorkflow();
      const workflowId = result.workflow_id;
      const workflowName = `工作流_${workflowId.substring(0, 8)}`;
      
      // 更新状态
      window.stateManager.setCurrentWorkflow(workflowId, workflowName);
      window.stateManager.addWorkflow({
        workflow_id: workflowId,
        name: workflowName
      });
      
      // 重置工作流相关状态
      window.stateManager.setState({
        workflowNodes: [],
        workflowConnections: [],
        selectedNode: null,
        nodeParameters: {},
        nodeTaskIds: {}
      });
      
      this.showAlert('工作流创建成功！');
    } catch (error) {
      this.showAlert(error.message);
    }
  }

  async handleAddTask() {
    const state = window.stateManager.getState();
    if (!state.currentWorkflowId) {
      this.showAlert('请先创建工作流');
      return;
    }

    try {
      // 调用API添加任务
      const result = await this.apiService.addTask(state.currentWorkflowId, 'code');
      const taskId = result.task_id;
      
      // 在画布中心添加一个空白任务节点
      const canvas = document.getElementById('workflow-canvas');
      const canvasRect = canvas.getBoundingClientRect();
      const centerX = canvasRect.width / 2;
      const centerY = canvasRect.height / 2;
      
      // 生成节点ID
      const nodeId = `node_${Date.now()}`;
      
      // 创建节点数据
      const newNode = {
        id: nodeId,
        taskId: taskId,
        type: 'code',
        name: `任务_${taskId.substring(0, 8)}`,
        x: centerX,
        y: centerY,
        status: 'pending',
        inputs: [],
        outputs: []
      };
      
      // 更新状态
      const currentNodes = state.workflowNodes || [];
      window.stateManager.setState({
        workflowNodes: [...currentNodes, newNode],
        nodeTaskIds: {
          ...state.nodeTaskIds,
          [nodeId]: taskId
        }
      });
      
      this.showAlert('任务添加成功！请在属性面板中配置任务详情。');
    } catch (error) {
      this.showAlert(`添加任务失败: ${error.message}`);
    }
  }

  async handleRunWorkflow() {
    const state = window.stateManager.getState();
    if (!state.currentWorkflowId) {
      this.showAlert('请先创建工作流');
      return;
    }

    try {
      // 设置运行状态
      window.stateManager.setState({ isSubmitting: true });
      
      // 调用API运行工作流
      await this.apiService.runWorkflow(state.currentWorkflowId);
      
      this.showAlert('工作流运行成功！');
    } catch (error) {
      this.showAlert(`运行失败: ${error.message}`);
    } finally {
      window.stateManager.setState({ isSubmitting: false });
    }
  }

  async handleSaveWorkflow() {
    const state = window.stateManager.getState();
    if (!state.currentWorkflowId) {
      this.showAlert('请先创建工作流');
      return;
    }

    try {
      // 保存所有连接
      for (const connection of state.workflowConnections) {
        const sourceTaskId = state.nodeTaskIds[connection.source];
        const targetTaskId = state.nodeTaskIds[connection.target];
        
        if (sourceTaskId && targetTaskId) {
          await this.apiService.addEdge(
            state.currentWorkflowId,
            sourceTaskId,
            targetTaskId
          );
        }
      }
      
      this.showAlert('工作流保存成功！');
    } catch (error) {
      this.showAlert(`保存失败: ${error.message}`);
    }
  }

  updateUI(state) {
    const createBtn = this.container.querySelector('#create-workflow-btn');
    const addTaskBtn = this.container.querySelector('#add-task-btn');
    const submitBtn = this.container.querySelector('#submit-workflow-btn');
    const saveBtn = this.container.querySelector('#save-workflow-btn');

    // 更新按钮状态
    addTaskBtn.disabled = !state.currentWorkflowId;
    submitBtn.disabled = !state.currentWorkflowId || state.isSubmitting;
    saveBtn.disabled = !state.currentWorkflowId;

    // 更新提交按钮文本
    if (state.isSubmitting) {
      submitBtn.innerHTML = '<span class="loading"><span class="spinner"></span> 运行中...</span>';
    } else {
      submitBtn.innerHTML = '<i class="fas fa-play"></i> 运行';
    }
  }

  showAlert(message, title = '提示') {
    // 创建提示模态框
    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.innerHTML = `
      <div class="modal-content" style="width: 400px;">
        <div class="modal-header">
          <h3>${title}</h3>
          <button id="close-modal" style="background: none; border: none; font-size: 20px; cursor: pointer;">&times;</button>
        </div>
        <div class="modal-body">
          <p>${message}</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" id="confirm-modal">确定</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // 绑定关闭事件
    const closeModal = () => {
      if (document.body.contains(modal)) {
        document.body.removeChild(modal);
      }
    };
    
    modal.querySelector('#close-modal').addEventListener('click', closeModal);
    modal.querySelector('#confirm-modal').addEventListener('click', closeModal);
    
    // 点击背景关闭
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeModal();
      }
    });
  }
}

// 导出组件
window.ToolbarComponent = ToolbarComponent;