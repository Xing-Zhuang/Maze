/**
 * 状态管理模块
 * 管理应用的整体状态
 */

class StateManager {
  constructor() {
    this.state = {
      // 工作流相关
      currentWorkflowId: null,
      currentWorkflowName: '未选择工作流',
      workflows: [],
      
      // 任务相关
      tasks: [], // 存储所有任务
      workflowNodes: [], // 工作流中的节点
      workflowConnections: [], // 工作流中的连接
      
      // UI状态
      selectedNode: null,
      connectingNode: null,
      isConnecting: false,
      connectingOutputParam: null,
      
      // 任务参数
      nodeParameters: {}, // 节点参数 {nodeId: {param1: value1, param2: value2}}
      nodeTaskIds: {}, // 节点ID到后端任务ID的映射
      
      // 运行相关
      runs: [],
      currentRunId: null,
      runResults: {},
      
      // 画布状态
      zoomLevel: 1,
      canvasOffset: { x: 0, y: 0 },
      
      // 其他状态
      isSubmitting: false
    };
    
    this.listeners = [];
  }

  /**
   * 获取当前状态
   */
  getState() {
    return { ...this.state };
  }

  /**
   * 更新状态
   * @param {Object} updates - 要更新的状态
   */
  setState(updates) {
    const oldState = { ...this.state };
    this.state = { ...this.state, ...updates };
    this.notifyListeners(oldState, this.state);
  }

  /**
   * 订阅状态变化
   * @param {Function} listener - 监听器函数
   */
  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * 通知所有监听器
   * @param {Object} oldState - 旧状态
   * @param {Object} newState - 新状态
   */
  notifyListeners(oldState, newState) {
    this.listeners.forEach(listener => {
      try {
        listener(newState, oldState);
      } catch (error) {
        console.error('状态监听器错误:', error);
      }
    });
  }

  // 工作流相关方法
  setCurrentWorkflow(workflowId, workflowName) {
    this.setState({
      currentWorkflowId: workflowId,
      currentWorkflowName: workflowName || '未命名工作流'
    });
  }

  addWorkflow(workflow) {
    const workflows = [...this.state.workflows, workflow];
    this.setState({ workflows });
  }

  // 任务相关方法
  addTask(task) {
    const tasks = [...this.state.tasks, task];
    this.setState({ tasks });
  }

  addWorkflowNode(node) {
    const workflowNodes = [...this.state.workflowNodes, node];
    this.setState({ workflowNodes });
  }

  removeWorkflowNode(nodeId) {
    const workflowNodes = this.state.workflowNodes.filter(node => node.id !== nodeId);
    const workflowConnections = this.state.workflowConnections.filter(
      conn => conn.source !== nodeId && conn.target !== nodeId
    );
    
    // 清理相关数据
    const nodeParameters = { ...this.state.nodeParameters };
    const nodeTaskIds = { ...this.state.nodeTaskIds };
    delete nodeParameters[nodeId];
    delete nodeTaskIds[nodeId];
    
    this.setState({
      workflowNodes,
      workflowConnections,
      nodeParameters,
      nodeTaskIds,
      selectedNode: this.state.selectedNode === nodeId ? null : this.state.selectedNode
    });
  }

  addWorkflowConnection(connection) {
    const workflowConnections = [...this.state.workflowConnections, connection];
    this.setState({ workflowConnections });
  }

  removeWorkflowConnection(sourceNodeId, targetNodeId) {
    const workflowConnections = this.state.workflowConnections.filter(
      conn => !(conn.source === sourceNodeId && conn.target === targetNodeId)
    );
    this.setState({ workflowConnections });
  }

  // UI状态方法
  selectNode(nodeId) {
    this.setState({ selectedNode: nodeId });
  }

  startConnecting(nodeId, outputParam) {
    this.setState({
      connectingNode: nodeId,
      isConnecting: true,
      connectingOutputParam: outputParam
    });
  }

  stopConnecting() {
    this.setState({
      connectingNode: null,
      isConnecting: false,
      connectingOutputParam: null
    });
  }

  // 参数相关方法
  setNodeParameter(nodeId, paramName, value) {
    const nodeParameters = { ...this.state.nodeParameters };
    if (!nodeParameters[nodeId]) {
      nodeParameters[nodeId] = {};
    }
    nodeParameters[nodeId][paramName] = value;
    this.setState({ nodeParameters });
  }

  setNodeTaskId(nodeId, taskId) {
    const nodeTaskIds = { ...this.state.nodeTaskIds };
    nodeTaskIds[nodeId] = taskId;
    this.setState({ nodeTaskIds });
  }

  // 运行相关方法
  addRun(run) {
    const runs = [...this.state.runs, run];
    this.setState({ runs });
  }

  updateRunStatus(runId, status) {
    const runs = this.state.runs.map(run => 
      run.run_id === runId ? { ...run, status } : run
    );
    this.setState({ runs });
  }

  setCurrentRun(runId) {
    this.setState({ currentRunId: runId });
  }

  setRunResults(runId, results) {
    const runResults = { ...this.state.runResults };
    runResults[runId] = results;
    this.setState({ runResults });
  }

  // 画布相关方法
  setZoomLevel(level) {
    this.setState({ zoomLevel: level });
  }

  setCanvasOffset(offset) {
    this.setState({ canvasOffset: offset });
  }

  // 其他方法
  setSubmitting(isSubmitting) {
    this.setState({ isSubmitting });
  }

  // 重置状态
  reset() {
    this.setState({
      currentWorkflowId: null,
      currentWorkflowName: '未选择工作流',
      workflowNodes: [],
      workflowConnections: [],
      selectedNode: null,
      connectingNode: null,
      isConnecting: false,
      connectingOutputParam: null,
      nodeParameters: {},
      nodeTaskIds: {},
      currentRunId: null,
      runResults: {},
      zoomLevel: 1,
      canvasOffset: { x: 0, y: 0 },
      isSubmitting: false
    });
  }
}

// 创建全局状态管理器实例
window.stateManager = new StateManager();
