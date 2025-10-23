# Maze Playground v2

这是 Maze 工作流框架的可视化界面组件，经过重构后采用模块化架构，适配新的后端 API。

## 项目结构

```
web/maze_playground/v2/
├── index.html              # 主入口文件
├── styles/
│   └── main.css           # 主样式文件
└── js/
    ├── main.js            # 应用入口文件
    ├── services/
    │   ├── api.js         # API服务层
    │   └── state.js       # 状态管理
    └── components/
        ├── toolbar.js     # 工具栏组件
        ├── sidebar.js     # 侧边栏组件
        ├── task-library.js # 任务库组件
        ├── workspace.js   # 工作区组件
        └── property-panel.js # 属性面板组件
```

## 主要特性

### 1. 模块化架构
- **API服务层**: 封装所有后端API调用
- **状态管理**: 集中管理应用状态
- **组件化**: 将UI拆分为独立的可复用组件

### 2. 适配新后端API
基于 `server.py` 中的API端点：
- `POST /create_workflow` - 创建工作流
- `POST /add_task` - 添加任务
- `POST /del_task` - 删除任务
- `POST /save_task` - 保存任务详情
- `POST /add_edge` - 添加边（任务依赖）
- `POST /del_edge` - 删除边
- `POST /run_workflow` - 运行工作流
- `WebSocket /get_workflow_res/{workflow_id}` - 获取运行结果

### 3. 核心组件

#### 工具栏 (ToolbarComponent)
- 创建工作流
- 运行工作流
- 保存工作流
- 设置输入参数

#### 侧边栏 (SidebarComponent)
- 任务库：提供可拖拽的任务模板
- 工作流列表：管理多个工作流
- 运行实例：查看运行历史

#### 工作区 (WorkspaceComponent)
- 可视化工作流设计器
- 拖拽创建节点
- 连接节点建立依赖关系
- 缩放和平移画布
- 执行结果展示

#### 属性面板 (PropertyPanelComponent)
- 编辑节点属性
- 设置任务参数
- 查看节点信息

### 4. 状态管理
使用 `StateManager` 类集中管理应用状态：
- 工作流状态
- 节点和连接信息
- UI状态（选中、连接中等）
- 运行状态和结果

## 使用方法

### 自定义任务工作流

1. 确保后端服务运行在 `http://localhost:8000`
2. 在浏览器中打开 `custom-task-test.html`
3. 点击"新建工作流"创建工作流
4. 点击"添加任务"添加空白任务节点到画布
5. 点击任务节点，在右侧属性面板配置任务：
   - **基本信息**: 设置任务名称
   - **输入参数**: 定义任务的输入参数（参数名、数据类型、输入模式、默认值）
   - **输出参数**: 定义任务的输出参数（参数名、数据类型）
   - **资源需求**: 设置CPU、内存、GPU等资源需求
   - **任务代码**: 编写Python代码实现任务逻辑
6. 点击"保存任务配置"保存任务详情
7. 重复步骤4-6添加更多任务
8. 连接任务建立依赖关系
9. 点击"运行"执行工作流

### 传统模板任务（待开发）

- 从左侧任务库拖拽预定义任务到画布
- 配置任务参数
- 建立任务连接
- 运行工作流

## 测试页面

- `index.html` - 主应用页面
- `api-init-test.html` - API初始化测试页面
- `layout-test.html` - 布局测试页面（包含调试信息）

## 技术栈

- **原生JavaScript**: 无框架依赖，轻量级
- **CSS3**: 现代化样式，支持响应式设计
- **Axios**: HTTP请求库
- **WebSocket**: 实时获取运行结果
- **Font Awesome**: 图标库

## 开发说明

### 添加新组件
1. 在 `js/components/` 目录下创建组件文件
2. 在 `index.html` 中添加脚本引用
3. 在 `main.js` 中初始化组件

### 扩展API服务
1. 在 `js/services/api.js` 中添加新的API方法
2. 确保错误处理和参数验证

### 修改状态管理
1. 在 `js/services/state.js` 中添加新的状态字段
2. 添加相应的getter/setter方法
3. 更新相关组件以响应状态变化

## 与原版本的差异

1. **简化API**: 移除了复杂的会话管理，直接使用工作流API
2. **模块化**: 将单一大文件拆分为多个模块
3. **组件化**: UI组件独立，便于维护和扩展
4. **状态管理**: 集中管理状态，避免数据不一致
5. **错误处理**: 统一的错误处理和用户提示

## 注意事项

- 确保后端服务正常运行
- 浏览器需要支持ES6语法
- 建议使用现代浏览器以获得最佳体验
- 如需修改API地址，请更新 `js/services/api.js` 中的 `baseUrl`

## 常见问题

### CORS错误
如果遇到 "Network Error" 或 CORS 相关错误，请确保后端服务已添加CORS支持。后端 `server.py` 需要包含：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### API测试
使用 `simple-test.html` 可以快速测试前端与后端的连接是否正常。
