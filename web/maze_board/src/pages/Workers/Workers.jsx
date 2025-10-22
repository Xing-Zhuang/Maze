import React from 'react';
import WorkerCard from './WorkerCard';

const Workers = ({ workers }) => {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Workers</h2>
      <div className="grid grid-cols-1 gap-4">
        {workers.map(worker => (
          <WorkerCard key={worker.worker_id} worker={worker} />
        ))}
      </div>
    </div>
  );
};

export default Workers;