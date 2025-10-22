import React, { useState } from 'react';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard/Dashboard';
import Workers from './pages/Workers/Workers';
import Workflows from './pages/Workflows/Workflows';
import WorkflowDetail from './pages/Workflows/WorkflowDetail';
import WorkflowRunDetail from './pages/Workflows/WorkflowRunDetail';
import { mockAPI, workflowRuns, runTaskExecutions } from './utils/mockData';

const App = () => {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [selectedWorkflow, setSelectedWorkflow] = useState(null);
  const [selectedRun, setSelectedRun] = useState(null);

  const handleWorkflowClick = (workflowId) => {
    setSelectedWorkflow(workflowId);
    setSelectedRun(null);
    setCurrentTab('workflow-detail');
  };

  const handleRunClick = (runId) => {
    setSelectedRun(runId);
    setCurrentTab('run-detail');
  };

  const handleBackToWorkflows = () => {
    setSelectedWorkflow(null);
    setSelectedRun(null);
    setCurrentTab('workflows');
  };

  const handleBackToWorkflowDetail = () => {
    setSelectedRun(null);
    setCurrentTab('workflow-detail');
  };

  const renderPage = () => {
    switch (currentTab) {
      case 'dashboard':
        return <Dashboard workers={mockAPI.workers} workflows={mockAPI.workflows} />;

      case 'workers':
        return <Workers workers={mockAPI.workers} />;

      case 'workflows':
        return <Workflows workflows={mockAPI.workflows} onWorkflowClick={handleWorkflowClick} />;

      case 'workflow-detail':
        if (!selectedWorkflow) return <Workflows workflows={mockAPI.workflows} onWorkflowClick={handleWorkflowClick} />;
        const workflow = mockAPI.workflows.find(w => w.workflow_id === selectedWorkflow);
        const runs = workflowRuns[selectedWorkflow] || [];
        return (
          <WorkflowDetail
            workflow={workflow}
            workflowDetails={mockAPI.workflowDetails[selectedWorkflow]}
            runs={runs}
            onRunClick={handleRunClick}
            onBack={handleBackToWorkflows}
          />
        );

      case 'run-detail':
        if (!selectedWorkflow || !selectedRun) return null;
        const run = workflowRuns[selectedWorkflow]?.find(r => r.run_id === selectedRun);
        const taskExecutions = runTaskExecutions[selectedRun] || [];
        return (
          <WorkflowRunDetail
            run={run}
            workflowDetails={mockAPI.workflowDetails[selectedWorkflow]}
            taskExecutions={taskExecutions}
            onBack={handleBackToWorkflowDetail}
          />
        );

      default:
        return <Dashboard workers={mockAPI.workers} workflows={mockAPI.workflows} />;
    }
  };

  return (
    <MainLayout currentTab={currentTab} onTabChange={setCurrentTab}>
      {renderPage()}
    </MainLayout>
  );
};

export default App;