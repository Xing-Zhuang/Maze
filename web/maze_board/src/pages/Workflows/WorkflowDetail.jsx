import React from 'react';
import WorkflowGraph from './components/WorkflowGraph';
import RunList from './components/RunList';

const WorkflowDetail = ({ workflow, workflowDetails, runs, onRunClick, onBack }) => {
  return (
    <div className="space-y-6">
      {/* 返回按钮 */}
      <button
        onClick={onBack}
        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
      >
        ← Back to Workflows
      </button>

      {/* 工作流信息 */}
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h2 className="text-2xl font-bold mb-2">{workflow.workflow_name}</h2>
        <div className="text-sm text-gray-400">
          <p>Workflow ID: {workflow.workflow_id}</p>
          <p>Created: {new Date(workflow.created_at).toLocaleString()}</p>
          <p>Total Runs: {workflow.total_runs}</p>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：静态 DAG 图 */}
        <div className="lg:col-span-2 bg-gray-800 p-6 rounded-lg border border-gray-700">
          <h3 className="text-xl font-semibold mb-4">Workflow Structure</h3>
          <p className="text-sm text-gray-400 mb-4">
            This is the static workflow definition. Click on a run to see execution details.
          </p>
          <WorkflowGraph
            nodes={workflowDetails?.nodes || []}
            edges={workflowDetails?.edges || []}
            isStatic={true}
          />
        </div>

        {/* 右侧：Run 列表 */}
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <h3 className="text-xl font-semibold mb-4">Run History</h3>
          <RunList runs={runs} onRunClick={onRunClick} />
        </div>
      </div>
    </div>
  );
};

export default WorkflowDetail;