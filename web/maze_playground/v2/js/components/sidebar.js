/**
 * 侧边栏组件
 */

class SidebarComponent {
  constructor(container) {
    this.container = container;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.className = 'sidebar';
    this.container.innerHTML = `
        <div id="task-library-container"></div>
        
        <div class="sidebar-section">
          <h3><i class="fas fa-project-diagram"></i> 工作流列表</h3>
          <div id="workflow-list"></div>
        </div>

        <div class="sidebar-section">
          <h3><i class="fas fa-running"></i> 运行实例</h3>
          <div id="run-list"></div>
        </div>
    `;
    
    // 初始化任务库组件
    this.taskLibrary = new TaskLibraryComponent(
      document.getElementById('task-library-container')
    );
  }

  bindEvents() {
    // 监听状态变化
    window.stateManager.subscribe((newState) => {
      this.updateUI(newState);
    });
  }

  updateUI(state) {
    this.renderWorkflowList(state.workflows, state.currentWorkflowId);
    this.renderRunList(state.runs, state.currentRunId);
  }

  renderWorkflowList(workflows, currentWorkflowId) {
    const workflowList = this.container.querySelector('#workflow-list');
    workflowList.innerHTML = '';

    workflows.forEach(workflow => {
      const workflowItem = document.createElement('div');
      workflowItem.className = 'workflow-item';
      if (workflow.workflow_id === currentWorkflowId) {
        workflowItem.classList.add('active');
      }
      workflowItem.innerHTML = `
        <h4>${workflow.name}</h4>
        <small>ID: ${workflow.workflow_id.substring(0, 12)}...</small>
      `;
      workflowItem.addEventListener('click', () => {
        this.loadWorkflow(workflow.workflow_id);
      });
      workflowList.appendChild(workflowItem);
    });
  }

  renderRunList(runs, currentRunId) {
    const runList = this.container.querySelector('#run-list');
    runList.innerHTML = '';

    runs.forEach(run => {
      const runItem = document.createElement('div');
      runItem.className = 'run-item';
      runItem.classList.add(run.status || 'running');
      if (run.run_id === currentRunId) {
        runItem.classList.add('active');
      }
      runItem.innerHTML = `
        <h4>运行实例</h4>
        <small>ID: ${run.run_id.substring(0, 12)}...</small>
        <div style="margin-top: 8px;">
          <span class="result-status ${run.status || 'running'}">${run.status || 'running'}</span>
        </div>
      `;
      runItem.addEventListener('click', () => {
        this.selectRun(run.run_id);
      });
      runList.appendChild(runItem);
    });
  }

  loadWorkflow(workflowId) {
    const state = window.stateManager.getState();
    const workflow = state.workflows.find(w => w.workflow_id === workflowId);
    
    if (workflow) {
      window.stateManager.setCurrentWorkflow(workflowId, workflow.name);
      
      // 重置工作流相关状态
      window.stateManager.setState({
        workflowNodes: [],
        workflowConnections: [],
        selectedNode: null,
        nodeParameters: {},
        nodeTaskIds: {}
      });
    }
  }

  selectRun(runId) {
    window.stateManager.setCurrentRun(runId);
    
    // 切换到结果选项卡
    const resultsTab = document.querySelector('.tab[data-tab="results"]');
    if (resultsTab) {
      resultsTab.click();
    }
  }
}

// 导出组件
window.SidebarComponent = SidebarComponent;
