import React from 'react';
import { CommunicationAnalysisView } from '../components/CommunicationAnalysisView';

export const CommunicationAnalysis: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12">
      <CommunicationAnalysisView initialRoomId="TEST001" />
    </div>
  );
};
