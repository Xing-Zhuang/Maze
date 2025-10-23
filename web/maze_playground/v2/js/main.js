/**
 * 主应用入口文件
 * 初始化所有组件并管理应用生命周期
 */

// 注意：在浏览器环境中，我们需要确保所有脚本都已加载
// 这里我们假设所有模块都通过script标签按顺序加载

class MazePlaygroundApp {
  constructor() {
    this.components = {};
    this.init();
  }

  init() {
    // 等待DOM加载完成
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.start());
    } else {
      this.start();
    }
  }

  start() {
    console.log('Maze Playground 应用启动中...');
    
    try {
      this.createAppStructure();
      this.initializeComponents();
      this.setupGlobalEventListeners();
      
      console.log('Maze Playground 应用启动完成');
    } catch (error) {
      console.error('应用启动失败:', error);
      this.showError('应用启动失败: ' + error.message);
    }
  }

  createAppStructure() {
    const appContainer = document.getElementById('app');
    if (!appContainer) {
      throw new Error('找不到应用容器 #app');
    }

    appContainer.innerHTML = `
      <div id="toolbar-container"></div>
      <div class="main-layout">
        <div id="sidebar-container"></div>
        <div id="workspace-container"></div>
        <div id="property-panel-container"></div>
      </div>
    `;
  }

  initializeComponents() {
    // 初始化全局服务
    window.api = new ApiService();
    window.stateManager = new StateManager();
    
    // 初始化工具栏
    this.components.toolbar = new ToolbarComponent(
      document.getElementById('toolbar-container')
    );

    // 初始化侧边栏
    this.components.sidebar = new SidebarComponent(
      document.getElementById('sidebar-container')
    );

    // 初始化工作区
    this.components.workspace = new WorkspaceComponent(
      document.getElementById('workspace-container')
    );

    // 初始化属性面板
    this.components.propertyPanel = new PropertyPanelComponent(
      document.getElementById('property-panel-container')
    );

    // 将工作区组件暴露到全局，供节点删除功能使用
    window.workspaceComponent = this.components.workspace;
  }

  setupGlobalEventListeners() {
    // 监听窗口大小变化
    window.addEventListener('resize', () => {
      this.handleResize();
    });

    // 监听页面卸载
    window.addEventListener('beforeunload', () => {
      this.handleBeforeUnload();
    });

    // 监听键盘快捷键
    document.addEventListener('keydown', (e) => {
      this.handleKeyboardShortcuts(e);
    });
  }

  handleResize() {
    // 重新计算画布大小等
    console.log('窗口大小变化，重新计算布局');
  }

  handleBeforeUnload() {
    // 清理资源
    console.log('页面即将卸载，清理资源');
  }

  handleKeyboardShortcuts(e) {
    // Ctrl+S 保存工作流
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      this.components.toolbar.handleSaveWorkflow();
    }
    
    // Ctrl+N 新建工作流
    if (e.ctrlKey && e.key === 'n') {
      e.preventDefault();
      this.components.toolbar.handleCreateWorkflow();
    }
    
    // Delete 删除选中节点
    if (e.key === 'Delete') {
      const state = window.stateManager.getState();
      if (state.selectedNode) {
        this.components.workspace.deleteNode(state.selectedNode);
      }
    }
  }

  showError(message) {
    // 创建错误提示
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ef4444;
      color: white;
      padding: 16px;
      border-radius: 8px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      z-index: 10000;
      max-width: 400px;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // 3秒后自动移除
    setTimeout(() => {
      if (document.body.contains(errorDiv)) {
        document.body.removeChild(errorDiv);
      }
    }, 3000);
  }

  // 获取应用状态（用于调试）
  getAppState() {
    return {
      components: Object.keys(this.components),
      state: window.stateManager.getState()
    };
  }

  // 重置应用（用于调试）
  resetApp() {
    window.stateManager.reset();
    console.log('应用已重置');
  }
}

// 启动应用
const app = new MazePlaygroundApp();

// 将应用实例暴露到全局，便于调试
window.mazeApp = app;

// 导出应用类（如果使用模块系统）
if (typeof module !== 'undefined' && module.exports) {
  module.exports = MazePlaygroundApp;
}
