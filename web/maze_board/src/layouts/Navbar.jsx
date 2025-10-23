import React from 'react';

const Navbar = ({ currentTab, onTabChange }) => {
  const tabs = [
    { id: 'dashboard', label: '📊 Dashboard' },
    { id: 'workers', label: '🖥️ Workers' },
    { id: 'workflows', label: '⚙️ Workflows' }
  ];

  return (
    <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <h1 className="text-3xl font-bold mb-4">Distributed Workflow Manager</h1>
      <div className="flex gap-4">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-6 py-2 rounded-lg font-medium transition ${
              currentTab === tab.id || 
              (tab.id === 'workflows' && currentTab === 'workflow-detail')
                ? 'bg-blue-600' 
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Navbar;