/**
 * 属性面板组件
 */

class PropertyPanelComponent {
  constructor(container) {
    this.container = container;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    this.container.className = 'property-panel';
    this.container.innerHTML = `
        <div class="property-header">
          <div class="property-title" id="property-title">属性面板</div>
          <div class="property-subtitle" id="property-subtitle">选择节点查看属性</div>
        </div>
        <div class="property-content" id="property-content">
          <p>请选择一个节点查看和编辑属性</p>
        </div>
    `;
  }

  bindEvents() {
    // 监听状态变化
    window.stateManager.subscribe((newState) => {
      this.updateUI(newState);
    });
  }

  updateUI(state) {
    if (state.selectedNode) {
      this.showNodeProperties(state.selectedNode, state);
    } else {
      this.showDefaultContent();
    }
  }

  showNodeProperties(nodeId, state) {
    const node = state.workflowNodes.find(n => n.id === nodeId);
    if (!node) return;

    this.container.querySelector('#property-title').textContent = '任务配置';
    this.container.querySelector('#property-subtitle').textContent = node.name;

    const nodeParams = state.nodeParameters[nodeId] || {};
    
    let paramsHtml = '<div class="task-config">';
    
    // 基本信息
    paramsHtml += `
      <div class="config-section">
        <h4><i class="fas fa-info-circle"></i> 基本信息</h4>
        <div class="param-item">
          <label class="param-label">任务名称</label>
          <input type="text" class="param-input" value="${node.name}" data-param="name">
        </div>
        <div class="param-item">
          <label class="param-label">任务ID</label>
          <input type="text" class="param-input" value="${state.nodeTaskIds[nodeId] || ''}" readonly>
        </div>
      </div>
    `;
    
    // 输入参数
    paramsHtml += `
      <div class="config-section">
        <h4><i class="fas fa-sign-in-alt"></i> 输入参数</h4>
        <div class="param-list" id="input-params-list">
          ${this.renderInputParams(nodeParams.inputParams || [])}
        </div>
        <button class="btn btn-sm btn-secondary" id="add-input-param">
          <i class="fas fa-plus"></i> 添加输入参数
        </button>
      </div>
    `;
    
    // 输出参数
    paramsHtml += `
      <div class="config-section">
        <h4><i class="fas fa-sign-out-alt"></i> 输出参数</h4>
        <div class="param-list" id="output-params-list">
          ${this.renderOutputParams(nodeParams.outputParams || [])}
        </div>
        <button class="btn btn-sm btn-secondary" id="add-output-param">
          <i class="fas fa-plus"></i> 添加输出参数
        </button>
      </div>
    `;
    
    // 资源需求
    paramsHtml += `
      <div class="config-section">
        <h4><i class="fas fa-server"></i> 资源需求</h4>
        <div class="param-item">
          <label class="param-label">CPU核心数</label>
          <input type="number" class="param-input" value="${nodeParams.cpu || 1}" data-param="cpu" min="0" step="0.1">
        </div>
        <div class="param-item">
          <label class="param-label">CPU内存(MB)</label>
          <input type="number" class="param-input" value="${nodeParams.cpu_mem || 128}" data-param="cpu_mem" min="0">
        </div>
        <div class="param-item">
          <label class="param-label">GPU数量</label>
          <input type="number" class="param-input" value="${nodeParams.gpu || 0}" data-param="gpu" min="0" step="0.1">
        </div>
        <div class="param-item">
          <label class="param-label">GPU内存(MB)</label>
          <input type="number" class="param-input" value="${nodeParams.gpu_mem || 0}" data-param="gpu_mem" min="0">
        </div>
      </div>
    `;
    
    // 任务代码
    paramsHtml += `
      <div class="config-section">
        <h4><i class="fas fa-code"></i> 任务代码</h4>
        <div class="param-item">
          <label class="param-label">Python代码</label>
          <textarea class="param-textarea" data-param="code_str" rows="10" placeholder="def task(params):&#10;    # 在这里编写任务逻辑&#10;    return {}">${nodeParams.code_str || ''}</textarea>
        </div>
      </div>
    `;
    
    // 保存按钮
    paramsHtml += `
      <div class="config-actions">
        <button class="btn btn-primary" id="save-task-config">
          <i class="fas fa-save"></i> 保存任务配置
        </button>
      </div>
    `;
    
    paramsHtml += '</div>';

    this.container.querySelector('#property-content').innerHTML = paramsHtml;

    // 绑定事件
    this.bindTaskConfigEvents(nodeId);
  }

  showDefaultContent() {
    this.container.querySelector('#property-title').textContent = '属性面板';
    this.container.querySelector('#property-subtitle').textContent = '选择节点查看属性';
    this.container.querySelector('#property-content').innerHTML = '<p>请选择一个节点查看和编辑属性</p>';
  }

  renderInputParams(inputParams) {
    if (!inputParams || inputParams.length === 0) {
      return '<p class="no-params">暂无输入参数</p>';
    }
    
    return inputParams.map((param, index) => `
      <div class="param-row" data-index="${index}">
        <div class="param-group">
          <label>参数名</label>
          <input type="text" value="${param.key || ''}" data-field="key" placeholder="参数名">
        </div>
        <div class="param-group">
          <label>数据类型</label>
          <select data-field="data_type">
            <option value="str" ${param.data_type === 'str' ? 'selected' : ''}>字符串</option>
            <option value="int" ${param.data_type === 'int' ? 'selected' : ''}>整数</option>
            <option value="float" ${param.data_type === 'float' ? 'selected' : ''}>浮点数</option>
            <option value="bool" ${param.data_type === 'bool' ? 'selected' : ''}>布尔值</option>
          </select>
        </div>
        <div class="param-group">
          <label>输入模式</label>
          <select data-field="input_schema">
            <option value="from_user" ${param.input_schema === 'from_user' ? 'selected' : ''}>用户输入</option>
            <option value="from_task" ${param.input_schema === 'from_task' ? 'selected' : ''}>任务输出</option>
          </select>
        </div>
        <div class="param-group">
          <label>默认值</label>
          <input type="text" value="${param.value || ''}" data-field="value" placeholder="默认值">
        </div>
        <button class="btn btn-sm btn-danger remove-param">
          <i class="fas fa-trash"></i>
        </button>
      </div>
    `).join('');
  }

  renderOutputParams(outputParams) {
    if (!outputParams || outputParams.length === 0) {
      return '<p class="no-params">暂无输出参数</p>';
    }
    
    return outputParams.map((param, index) => `
      <div class="param-row" data-index="${index}">
        <div class="param-group">
          <label>参数名</label>
          <input type="text" value="${param.key || ''}" data-field="key" placeholder="参数名">
        </div>
        <div class="param-group">
          <label>数据类型</label>
          <select data-field="data_type">
            <option value="str" ${param.data_type === 'str' ? 'selected' : ''}>字符串</option>
            <option value="int" ${param.data_type === 'int' ? 'selected' : ''}>整数</option>
            <option value="float" ${param.data_type === 'float' ? 'selected' : ''}>浮点数</option>
            <option value="bool" ${param.data_type === 'bool' ? 'selected' : ''}>布尔值</option>
          </select>
        </div>
        <button class="btn btn-sm btn-danger remove-param">
          <i class="fas fa-trash"></i>
        </button>
      </div>
    `).join('');
  }

  bindTaskConfigEvents(nodeId) {
    // 绑定基本参数变化
    this.container.querySelectorAll('.param-input, .param-textarea').forEach(input => {
      input.addEventListener('change', (e) => {
        this.handleParameterChange(nodeId, e.target);
      });
    });

    // 添加输入参数
    this.container.querySelector('#add-input-param').addEventListener('click', () => {
      this.addInputParam(nodeId);
    });

    // 添加输出参数
    this.container.querySelector('#add-output-param').addEventListener('click', () => {
      this.addOutputParam(nodeId);
    });

    // 删除参数
    this.container.querySelectorAll('.remove-param').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.removeParam(nodeId, e.target.closest('.param-row'));
      });
    });

    // 保存任务配置
    this.container.querySelector('#save-task-config').addEventListener('click', () => {
      this.saveTaskConfig(nodeId);
    });
  }

  addInputParam(nodeId) {
    const state = window.stateManager.getState();
    const nodeParams = state.nodeParameters[nodeId] || {};
    const inputParams = nodeParams.inputParams || [];
    
    inputParams.push({
      key: '',
      data_type: 'str',
      input_schema: 'from_user',
      value: ''
    });
    
    window.stateManager.setNodeParameter(nodeId, 'inputParams', inputParams);
    this.updateUI(state);
  }

  addOutputParam(nodeId) {
    const state = window.stateManager.getState();
    const nodeParams = state.nodeParameters[nodeId] || {};
    const outputParams = nodeParams.outputParams || [];
    
    outputParams.push({
      key: '',
      data_type: 'str'
    });
    
    window.stateManager.setNodeParameter(nodeId, 'outputParams', outputParams);
    this.updateUI(state);
  }

  removeParam(nodeId, paramRow) {
    const paramType = paramRow.closest('.config-section').querySelector('h4').textContent.includes('输入') ? 'inputParams' : 'outputParams';
    const index = parseInt(paramRow.getAttribute('data-index'));
    
    const state = window.stateManager.getState();
    const nodeParams = state.nodeParameters[nodeId] || {};
    const params = nodeParams[paramType] || [];
    
    params.splice(index, 1);
    window.stateManager.setNodeParameter(nodeId, paramType, params);
    this.updateUI(state);
  }

  async saveTaskConfig(nodeId) {
    const state = window.stateManager.getState();
    const node = state.workflowNodes.find(n => n.id === nodeId);
    const nodeParams = state.nodeParameters[nodeId] || {};
    const taskId = state.nodeTaskIds[nodeId];
    
    if (!taskId) {
      alert('任务ID不存在，请重新添加任务');
      return;
    }

    try {
      // 构建任务输入
      const taskInput = {
        input_params: {}
      };
      
      (nodeParams.inputParams || []).forEach((param, index) => {
        if (param.key) {
          taskInput.input_params[index + 1] = {
            key: param.key,
            input_schema: param.input_schema || 'from_user',
            data_type: param.data_type || 'str',
            value: param.value || ''
          };
        }
      });

      // 构建任务输出
      const taskOutput = {
        output_params: {}
      };
      
      (nodeParams.outputParams || []).forEach((param, index) => {
        if (param.key) {
          taskOutput.output_params[index + 1] = {
            key: param.key,
            data_type: param.data_type || 'str'
          };
        }
      });

      // 构建资源需求
      const resources = {
        cpu: parseFloat(nodeParams.cpu) || 1,
        cpu_mem: parseInt(nodeParams.cpu_mem) || 128,
        gpu: parseFloat(nodeParams.gpu) || 0,
        gpu_mem: parseInt(nodeParams.gpu_mem) || 0
      };

      // 获取代码
      const codeStr = nodeParams.code_str || `def task(params):
    # 在这里编写任务逻辑
    return {}`;

      // 调用API保存任务
      await window.api.saveTask(
        state.currentWorkflowId,
        taskId,
        taskInput,
        taskOutput,
        resources,
        codeStr
      );

      alert('任务配置保存成功！');
    } catch (error) {
      alert(`保存失败: ${error.message}`);
    }
  }

  handleParameterChange(nodeId, input) {
    const paramName = input.getAttribute('data-param');
    const paramValue = input.value;

    if (paramName === 'name') {
      // 更新节点名称
      const state = window.stateManager.getState();
      const nodeIndex = state.workflowNodes.findIndex(n => n.id === nodeId);
      if (nodeIndex !== -1) {
        const updatedNodes = [...state.workflowNodes];
        updatedNodes[nodeIndex] = { ...updatedNodes[nodeIndex], name: paramValue };
        window.stateManager.setState({ workflowNodes: updatedNodes });
      }
    } else {
      // 更新节点参数
      window.stateManager.setNodeParameter(nodeId, paramName, paramValue);
    }
  }
}

// 导出组件
window.PropertyPanelComponent = PropertyPanelComponent;
