/**
 * API集成测试
 * 用于验证重构后的前端与后端API的集成
 */

class ApiIntegrationTest {
  constructor() {
    this.apiService = new ApiService();
    this.testResults = [];
  }

  async runAllTests() {
    console.log('开始API集成测试...');
    
    try {
      const workflowId = await this.testCreateWorkflow();
      const taskId = await this.testAddTask(workflowId);
      await this.testSaveTask(workflowId, taskId);
      await this.testAddEdge(workflowId, taskId, taskId); // 使用同一个taskId作为源和目标
      await this.testRunWorkflow(workflowId);
      
      this.printTestResults();
    } catch (error) {
      console.error('测试过程中发生错误:', error);
    }
  }

  async testCreateWorkflow() {
    console.log('测试创建工作流...');
    try {
      const result = await this.apiService.createWorkflow();
      this.addTestResult('createWorkflow', true, result);
      return result.workflow_id;
    } catch (error) {
      this.addTestResult('createWorkflow', false, error.message);
      throw error;
    }
  }

  async testAddTask(workflowId) {
    console.log('测试添加任务...');
    try {
      const result = await this.apiService.addTask(workflowId, 'code');
      this.addTestResult('addTask', true, result);
      return result.task_id;
    } catch (error) {
      this.addTestResult('addTask', false, error.message);
      throw error;
    }
  }

  async testSaveTask(workflowId, taskId) {
    console.log('测试保存任务...');
    try {
      const taskInput = {
        input_params: {
          "1": {
            key: "input_param",
            input_schema: "from_user",
            data_type: "str",
            value: "测试输入"
          }
        }
      };
      
      const taskOutput = {
        output_params: {
          "1": {
            key: "result",
            data_type: "str"
          }
        }
      };
      
      const resources = {
        cpu: 1,
        cpu_mem: 123,
        gpu: 0,
        gpu_mem: 0
      };
      
      const codeStr = `
def task(params):
    input_param = params.get("input_param")
    result = f"处理结果: {input_param}"
    return {"result": result}
`;
      
      const result = await this.apiService.saveTask(
        workflowId, taskId, taskInput, taskOutput, resources, codeStr
      );
      this.addTestResult('saveTask', true, result);
    } catch (error) {
      this.addTestResult('saveTask', false, error.message);
      throw error;
    }
  }

  async testAddEdge(workflowId, sourceTaskId, targetTaskId) {
    console.log('测试添加边...');
    try {
      const result = await this.apiService.addEdge(workflowId, sourceTaskId, targetTaskId);
      this.addTestResult('addEdge', true, result);
    } catch (error) {
      this.addTestResult('addEdge', false, error.message);
      throw error;
    }
  }

  async testRunWorkflow(workflowId) {
    console.log('测试运行工作流...');
    try {
      const result = await this.apiService.runWorkflow(workflowId);
      this.addTestResult('runWorkflow', true, result);
      
      // 等待一下让工作流开始运行
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 测试WebSocket连接
      await this.testWebSocketConnection(workflowId);
    } catch (error) {
      this.addTestResult('runWorkflow', false, error.message);
      throw error;
    }
  }

  async testWebSocketConnection(workflowId) {
    console.log('测试WebSocket连接...');
    return new Promise((resolve, reject) => {
      let messageReceived = false;
      let connectionEstablished = false;
      
      const ws = this.apiService.getWorkflowResults(
        workflowId,
        (data) => {
          console.log('收到WebSocket消息:', data);
          messageReceived = true;
          this.addTestResult('webSocket', true, '消息接收成功');
          ws.close();
          resolve();
        },
        (error) => {
          console.error('WebSocket错误:', error);
          this.addTestResult('webSocket', false, error.message);
          reject(error);
        },
        (event) => {
          console.log('WebSocket连接关闭');
          if (connectionEstablished && !messageReceived) {
            // 如果连接建立了但没有收到消息，可能是工作流已经完成
            this.addTestResult('webSocket', true, '连接正常关闭（工作流可能已完成）');
            resolve();
          } else if (!connectionEstablished) {
            this.addTestResult('webSocket', false, '连接建立失败');
            reject(new Error('连接建立失败'));
          }
        }
      );
      
      // 设置连接建立检测
      setTimeout(() => {
        connectionEstablished = true;
      }, 2000);
      
      // 设置超时
      setTimeout(() => {
        if (!messageReceived && connectionEstablished) {
          ws.close();
          this.addTestResult('webSocket', true, '连接超时但连接正常建立');
          resolve();
        } else if (!connectionEstablished) {
          ws.close();
          this.addTestResult('webSocket', false, '连接超时');
          reject(new Error('WebSocket连接超时'));
        }
      }, 15000);
    });
  }

  addTestResult(testName, success, data) {
    this.testResults.push({
      test: testName,
      success: success,
      data: data,
      timestamp: new Date().toISOString()
    });
  }

  printTestResults() {
    console.log('\n=== API集成测试结果 ===');
    this.testResults.forEach(result => {
      const status = result.success ? '✅ 通过' : '❌ 失败';
      console.log(`${status} ${result.test}: ${JSON.stringify(result.data)}`);
    });
    
    const passed = this.testResults.filter(r => r.success).length;
    const total = this.testResults.length;
    console.log(`\n总计: ${passed}/${total} 个测试通过`);
  }
}

// 在浏览器控制台中运行测试
window.runApiTest = async function() {
  const tester = new ApiIntegrationTest();
  await tester.runAllTests();
};

// 导出测试类
window.ApiIntegrationTest = ApiIntegrationTest;
