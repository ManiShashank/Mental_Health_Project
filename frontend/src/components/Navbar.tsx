import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, BookOpen, Layers, ShieldAlert, Cpu, MessageSquare } from 'lucide-react';
import { HealthBadge } from './HealthBadge';

export const Navbar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { name: 'Overview', path: '/', icon: Layers },
    { name: 'Live Chat', path: '/chat', icon: MessageSquare },
    { name: 'M3 Analysis', path: '/analysis', icon: Activity },
    { name: 'System Status', path: '/status', icon: Cpu },
    { name: 'Documentation', path: '/docs', icon: BookOpen },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-950/80 border-b border-slate-800">
      {/* Top Advisory Banner */}
      <div className="bg-amber-950/40 border-b border-amber-800/30 px-4 py-1.5 text-center text-xs text-amber-300/90 font-medium flex items-center justify-center gap-2">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
        <span>
          <strong>Academic Research Prototype:</strong> Analytical pre-screening tool only. Not intended for clinical diagnosis or medical evaluation.
        </span>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center shadow-lg shadow-emerald-950/50 group-hover:scale-105 transition-transform">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="font-bold text-sm sm:text-base tracking-tight text-white flex items-center gap-2">
                <span>MultiModal Pre-Screening</span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  M1 Foundation
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-normal hidden sm:block">
                Personalized Communication & Language Pattern Analysis
              </p>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1 sm:gap-3">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}

            <div className="h-5 w-px bg-slate-800 mx-1 hidden sm:block" />

            <HealthBadge />
          </nav>
        </div>
      </div>
    </header>
  );
};
