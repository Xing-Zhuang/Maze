/**
 * 工作区组件
 */

class WorkspaceComponent {
  constructor(container) {
    this.container = container;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.className = 'workspace';
    this.container.innerHTML = `
        <div class="workspace-header">
          <div class="workflow-info">
            <div class="workflow-title" id="current-workflow-name">未选择工作流</div>
            <div class="workflow-stats">
              <div class="stat-item">
                <i class="fas fa-cube"></i>
                <span id="node-count">0</span> 节点
              </div>
              <div class="stat-item">
                <i class="fas fa-link"></i>
                <span id="connection-count">0</span> 连接
              </div>
            </div>
          </div>
          <div class="workflow-actions">
            <button class="btn btn-secondary btn-sm" id="zoom-in">
              <i class="fas fa-search-plus"></i>
            </button>
            <button class="btn btn-secondary btn-sm" id="zoom-out">
              <i class="fas fa-search-minus"></i>
            </button>
            <button class="btn btn-secondary btn-sm" id="fit-view">
              <i class="fas fa-expand"></i>
            </button>
          </div>
        </div>

        <div class="tabs">
          <div class="tab active" data-tab="designer">工作流设计器</div>
          <div class="tab" data-tab="results">执行结果</div>
        </div>

        <div class="tab-content active" id="designer-tab">
          <div class="canvas-container">
            <div class="canvas" id="workflow-canvas">
              <div class="drag-hint" id="drag-hint">
                <i class="fas fa-project-diagram"></i>
                <p>拖拽任务到此处创建工作流</p>
                <small>从左侧工具箱中选择任务并拖拽到此处</small>
              </div>
            </div>
          </div>
        </div>

        <div class="tab-content" id="results-tab">
          <div class="results-container" id="results-container">
            <p>请选择一个运行实例查看结果</p>
          </div>
        </div>
    `;
  }

  bindEvents() {
    // 选项卡切换
    this.container.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        this.switchTab(tab.getAttribute('data-tab'));
      });
    });

    // 缩放功能
    this.container.querySelector('#zoom-in').addEventListener('click', () => {
      this.zoomIn();
    });

    this.container.querySelector('#zoom-out').addEventListener('click', () => {
      this.zoomOut();
    });

    this.container.querySelector('#fit-view').addEventListener('click', () => {
      this.fitView();
    });

    // 画布拖拽
    this.setupCanvasDragAndDrop();

    // 监听状态变化
    window.stateManager.subscribe((newState) => {
      this.updateUI(newState);
    });
  }

  switchTab(tabName) {
    // 更新选项卡状态
    this.container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    this.container.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    this.container.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    this.container.querySelector(`#${tabName}-tab`).classList.add('active');

    // 如果切换到结果选项卡，加载结果
    if (tabName === 'results') {
      const state = window.stateManager.getState();
      if (state.currentRunId) {
        this.loadRunResults(state.currentRunId);
      }
    }
  }

  zoomIn() {
    const state = window.stateManager.getState();
    const newZoomLevel = Math.min(state.zoomLevel + 0.1, 2);
    window.stateManager.setZoomLevel(newZoomLevel);
    this.updateCanvasTransform();
  }

  zoomOut() {
    const state = window.stateManager.getState();
    const newZoomLevel = Math.max(state.zoomLevel - 0.1, 0.5);
    window.stateManager.setZoomLevel(newZoomLevel);
    this.updateCanvasTransform();
  }

  fitView() {
    window.stateManager.setZoomLevel(1);
    this.updateCanvasTransform();
  }

  updateCanvasTransform() {
    const state = window.stateManager.getState();
    const canvas = this.container.querySelector('#workflow-canvas');
    canvas.style.transform = `scale(${state.zoomLevel})`;
  }

  setupCanvasDragAndDrop() {
    const canvas = this.container.querySelector('#workflow-canvas');

    // 拖拽任务到画布
    canvas.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    canvas.addEventListener('drop', (e) => {
      e.preventDefault();
      this.handleTaskDrop(e);
    });

    // 节点连接
    canvas.addEventListener('click', (e) => {
      this.handleConnectorClick(e);
    });

    // 节点选择
    canvas.addEventListener('click', (e) => {
      this.handleNodeClick(e);
    });
  }

  handleTaskDrop(e) {
    const state = window.stateManager.getState();
    if (!state.currentWorkflowId) {
      this.showAlert('请先创建工作流');
      return;
    }

    const taskData = e.dataTransfer.getData('task');
    if (taskData) {
      const task = JSON.parse(taskData);
      this.createWorkflowNode(task, e.offsetX, e.offsetY);
    }
  }

  createWorkflowNode(task, x, y) {
    const nodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const newNode = {
      id: nodeId,
      name: task.name || '新任务',
      function_name: task.name || 'task',
      task_type: task.task_type || 'code',
      x: x - 120,
      y: y - 70
    };

    window.stateManager.addWorkflowNode(newNode);
    this.showAlert(`已将任务 "${task.name}" 添加到工作流中`);
  }

  handleConnectorClick(e) {
    if (e.target.classList.contains('node-connector')) {
      const nodeId = e.target.getAttribute('data-node-id');
      const inputParam = e.target.getAttribute('data-param-name');
      const outputParam = e.target.getAttribute('data-output-param');

      if (e.target.classList.contains('connector-source')) {
        this.handleSourceConnectorClick(nodeId, outputParam, e);
      } else if (e.target.classList.contains('connector-target')) {
        this.handleTargetConnectorClick(nodeId, inputParam);
      }
    }
  }

  handleSourceConnectorClick(nodeId, outputParam, e) {
    const state = window.stateManager.getState();
    
    if (!state.isConnecting) {
      window.stateManager.startConnecting(nodeId, outputParam);
      e.target.style.background = '#f59e0b';
    } else {
      window.stateManager.stopConnecting();
      e.target.style.background = '#10b981';
    }
  }

  handleTargetConnectorClick(nodeId, inputParam) {
    const state = window.stateManager.getState();
    
    if (state.isConnecting && state.connectingNode && state.connectingNode !== nodeId) {
      // 创建连接
      const newConnection = {
        source: state.connectingNode,
        target: nodeId,
        inputParam: inputParam,
        outputParam: state.connectingOutputParam
      };
      
      window.stateManager.addWorkflowConnection(newConnection);
      window.stateManager.stopConnecting();
      
      // 重置连接点样式
      const connectingElement = document.querySelector(`[data-node-id="${state.connectingNode}"].connector-source`);
      if (connectingElement) {
        connectingElement.style.background = '#10b981';
      }
    } else {
      window.stateManager.stopConnecting();
    }
  }

  handleNodeClick(e) {
    if (e.target.classList.contains('node-delete')) {
      const nodeId = e.target.getAttribute('onclick').match(/'([^']+)'/)[1];
      this.deleteNode(nodeId);
    } else if (!e.target.classList.contains('node-connector') && 
               !e.target.classList.contains('param-input')) {
      const nodeElement = e.target.closest('.workflow-node');
      if (nodeElement) {
        const nodeId = nodeElement.id.replace('node-', '');
        window.stateManager.selectNode(nodeId);
      }
    }
  }

  deleteNode(nodeId) {
    window.stateManager.removeWorkflowNode(nodeId);
  }

  loadRunResults(runId) {
    const state = window.stateManager.getState();
    const results = state.runResults[runId];
    
    const resultsContainer = this.container.querySelector('#results-container');
    
    if (results) {
      resultsContainer.innerHTML = `
        <h4>运行结果</h4>
        <div class="result-item">
          <div class="result-header">
            <div class="result-title">运行 ${runId.substring(0, 8)}...</div>
            <div class="result-status ${results.status || 'running'}">${results.status || 'running'}</div>
          </div>
          <div class="result-content">${JSON.stringify(results, null, 2)}</div>
        </div>
      `;
    } else {
      resultsContainer.innerHTML = '<p>正在加载结果...</p>';
    }
  }

  updateUI(state) {
    // 更新工作流信息
    this.container.querySelector('#current-workflow-name').textContent = state.currentWorkflowName;
    this.container.querySelector('#node-count').textContent = state.workflowNodes.length;
    this.container.querySelector('#connection-count').textContent = state.workflowConnections.length;

    // 更新画布
    this.renderWorkflowCanvas(state);
    
    // 更新拖拽提示
    const dragHint = this.container.querySelector('#drag-hint');
    dragHint.style.display = state.workflowNodes.length > 0 ? 'none' : 'block';
  }

  renderWorkflowCanvas(state) {
    const canvas = this.container.querySelector('#workflow-canvas');
    
    // 清除现有节点和连接
    const oldNodes = canvas.querySelectorAll('.workflow-node, .connection-line, .connection-arrow');
    oldNodes.forEach(node => node.remove());

    // 渲染节点
    state.workflowNodes.forEach(node => {
      const nodeElement = this.createNodeElement(node, state);
      canvas.appendChild(nodeElement);
    });

    // 渲染连接
    state.workflowConnections.forEach(conn => {
      this.drawConnection(conn, state);
    });
  }

  createNodeElement(node, state) {
    const nodeElement = document.createElement('div');
    nodeElement.className = 'workflow-node';
    nodeElement.id = `node-${node.id}`;
    nodeElement.style.left = `${node.x}px`;
    nodeElement.style.top = `${node.y}px`;

    if (node.id === state.selectedNode) {
      nodeElement.classList.add('selected');
    }

    nodeElement.innerHTML = `
      <div class="node-header">
        <div class="node-title">${node.name}</div>
        <div class="node-type">${node.task_type || 'task'}</div>
      </div>
      <div class="node-function">${node.function_name}</div>
      <div class="node-delete" onclick="workspaceComponent.deleteNode('${node.id}')">×</div>
      <div class="node-connector connector-source" data-node-id="${node.id}" data-output-param="result"></div>
      <div class="node-connector connector-target" data-node-id="${node.id}" data-param-name="input"></div>
    `;

    // 使节点可拖拽
    this.makeNodeDraggable(nodeElement, node);

    return nodeElement;
  }

  makeNodeDraggable(nodeElement, node) {
    let isDragging = false;
    let offsetX, offsetY;

    nodeElement.addEventListener('mousedown', (e) => {
      if (e.target.classList.contains('node-connector')) return;
      if (e.target.classList.contains('node-delete')) return;

      isDragging = true;
      const rect = nodeElement.getBoundingClientRect();
      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;
      nodeElement.style.zIndex = '100';
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;

      const canvasRect = this.container.querySelector('#workflow-canvas').getBoundingClientRect();
      let x = e.clientX - canvasRect.left - offsetX;
      let y = e.clientY - canvasRect.top - offsetY;

      x = Math.max(0, Math.min(x, canvasRect.width - 240));
      y = Math.max(0, Math.min(y, canvasRect.height - 140));

      nodeElement.style.left = `${x}px`;
      nodeElement.style.top = `${y}px`;
      node.x = x;
      node.y = y;
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
      nodeElement.style.zIndex = '10';
    });
  }

  drawConnection(conn, state) {
    const sourceNode = state.workflowNodes.find(n => n.id === conn.source);
    const targetNode = state.workflowNodes.find(n => n.id === conn.target);

    if (!sourceNode || !targetNode) return;

    const canvas = this.container.querySelector('#workflow-canvas');
    const sourceElement = document.getElementById(`node-${conn.source}`);
    const targetElement = document.getElementById(`node-${conn.target}`);

    if (!sourceElement || !targetElement) return;

    const sourceRect = sourceElement.getBoundingClientRect();
    const targetRect = targetElement.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();

    const sourceX = sourceRect.left + sourceRect.width - canvasRect.left;
    const sourceY = sourceRect.top + sourceRect.height/2 - canvasRect.top;
    const targetX = targetRect.left - canvasRect.left;
    const targetY = targetRect.top + targetRect.height/2 - canvasRect.top;

    const length = Math.sqrt(Math.pow(targetX - sourceX, 2) + Math.pow(targetY - sourceY, 2));
    const angle = Math.atan2(targetY - sourceY, targetX - sourceX) * 180 / Math.PI;

    const line = document.createElement('div');
    line.className = 'connection-line';
    line.style.width = `${length}px`;
    line.style.left = `${sourceX}px`;
    line.style.top = `${sourceY}px`;
    line.style.transform = `rotate(${angle}deg)`;

    const arrow = document.createElement('div');
    arrow.className = 'connection-arrow';
    arrow.style.left = `${targetX}px`;
    arrow.style.top = `${targetY}px`;
    arrow.style.transform = `rotate(${angle}deg)`;

    canvas.appendChild(line);
    canvas.appendChild(arrow);
  }

  showAlert(message, title = '提示') {
    // 创建提示模态框
    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.innerHTML = `
      <div class="modal-content" style="width: 400px;">
        <div class="modal-header">
          <h3>${title}</h3>
          <button id="close-alert" style="background: none; border: none; font-size: 20px; cursor: pointer;">&times;</button>
        </div>
        <div class="modal-body" style="padding: 24px; text-align: center;">
          <p style="margin: 0; line-height: 1.6;">${message}</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" id="confirm-alert">确定</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // 绑定关闭事件
    const closeModal = () => {
      document.body.removeChild(modal);
    };
    
    modal.querySelector('#close-alert').addEventListener('click', closeModal);
    modal.querySelector('#confirm-alert').addEventListener('click', closeModal);
    
    // 点击外部关闭
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeModal();
      }
    });
  }
}

// 导出组件
window.WorkspaceComponent = WorkspaceComponent;
