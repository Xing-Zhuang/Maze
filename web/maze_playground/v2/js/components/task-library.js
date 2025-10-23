/**
 * 任务库组件
 * 提供可拖拽的任务模板
 */

class TaskLibraryComponent {
  constructor(container) {
    this.container = container;
    this.tasks = this.getDefaultTasks();
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  getDefaultTasks() {
    return [
      {
        name: 'print_task',
        description: '打印任务 - 输出文本信息',
        task_type: 'code'
      },
      {
        name: 'math_task',
        description: '数学计算任务 - 执行基本运算',
        task_type: 'code'
      },
      {
        name: 'data_task',
        description: '数据处理任务 - 处理和分析数据',
        task_type: 'code'
      },
      {
        name: 'file_task',
        description: '文件操作任务 - 读写文件',
        task_type: 'code'
      }
    ];
  }

  render() {
    this.container.innerHTML = `
      <div class="sidebar-section">
        <h3><i class="fas fa-toolbox"></i> 可用任务</h3>
        <div class="search-box">
          <i class="fas fa-search"></i>
          <input type="text" placeholder="搜索任务..." id="task-search">
        </div>
        <div id="task-list"></div>
      </div>
    `;
  }

  bindEvents() {
    // 搜索功能
    const searchInput = this.container.querySelector('#task-search');
    searchInput.addEventListener('input', (e) => {
      this.filterTasks(e.target.value);
    });
  }

  filterTasks(searchTerm) {
    const filteredTasks = this.tasks.filter(task => 
      task.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      task.description.toLowerCase().includes(searchTerm.toLowerCase())
    );
    this.renderTaskList(filteredTasks);
  }

  renderTaskList(tasks = this.tasks) {
    const taskList = this.container.querySelector('#task-list');
    taskList.innerHTML = '';

    tasks.forEach(task => {
      const taskItem = document.createElement('div');
      taskItem.className = 'task-item';
      taskItem.draggable = true;
      taskItem.innerHTML = `
        <div class="task-header">
          <div class="task-icon">
            <i class="fas fa-cog"></i>
          </div>
          <div class="task-name">${task.name}</div>
        </div>
        <div class="task-description">${task.description}</div>
        <div class="task-meta">
          <span>类型: ${task.task_type}</span>
          <span>ID: ${task.name.substring(0, 8)}</span>
        </div>
      `;
      
      taskItem.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('task', JSON.stringify(task));
      });
      
      taskList.appendChild(taskItem);
    });
  }
}

// 导出组件
window.TaskLibraryComponent = TaskLibraryComponent;
