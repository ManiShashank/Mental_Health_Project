import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Home } from './pages/Home';
import { ChatRoom } from './pages/ChatRoom';
import { CommunicationAnalysis } from './pages/CommunicationAnalysis';
import { SystemStatus } from './pages/SystemStatus';
import { Documentation } from './pages/Documentation';

export const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-emerald-500 selection:text-white">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chat" element={<ChatRoom />} />
            <Route path="/analysis" element={<CommunicationAnalysis />} />
            <Route path="/status" element={<SystemStatus />} />
            <Route path="/docs" element={<Documentation />} />
            {/* Fallback */}
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
};

export default App;
