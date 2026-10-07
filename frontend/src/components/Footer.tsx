import React from 'react';
import { ShieldCheck, GitBranch, Terminal } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800 bg-slate-950/60 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-xs text-slate-400">
          <div>
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm mb-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Research Protocol & Ethics</span>
            </div>
            <p className="leading-relaxed">
              This platform measures statistical deviations against an individual's historical baseline. It operates strictly within non-diagnostic boundaries to support academic computational linguistics and acoustic research.
            </p>
          </div>

          <div>
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm mb-2">
              <GitBranch className="w-4 h-4 text-indigo-400" />
              <span>Architecture & Status</span>
            </div>
            <p className="leading-relaxed">
              <strong>Milestone 1: Project Foundation.</strong> Clean separation of React frontend, FastAPI backend routing, Pydantic settings, and comprehensive documentation suite.
            </p>
          </div>

          <div>
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm mb-2">
              <Terminal className="w-4 h-4 text-teal-400" />
              <span>Technology Stack</span>
            </div>
            <ul className="space-y-1 font-mono text-[11px] text-slate-400">
              <li>• Backend: Python 3.13 + FastAPI + Pydantic v2</li>
              <li>• Frontend: React 18 + TypeScript + Vite + Tailwind</li>
              <li>• API Spec: OpenAPI 3.1 & RESTful Endpoints</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-500">
          <p>© 2026 Academic Research Project. Analytical Pre-Screening System.</p>
          <div className="flex items-center gap-4">
            <span className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              FastAPI Core v1.0.0
            </span>
            <span>•</span>
            <span>REST API /api/v1</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
