import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, UploadCloud, AlertTriangle, Cpu, FileText, Settings } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/live', label: 'Live Detection', icon: Radio },
  { path: '/analyze', label: 'Upload Analysis', icon: UploadCloud },
  { path: '/incidents', label: 'Incident Center', icon: AlertTriangle },
  { path: '/model', label: 'Model Center', icon: Cpu },
  { path: '/reports', label: 'Reports', icon: FileText },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  return (
    <aside className="w-64 border-r border-border bg-surface/50 backdrop-blur flex flex-col hidden md:flex">
      <div className="h-16 flex items-center px-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-glow-primary flex items-center justify-center font-bold text-white shadow-lg">V</div>
          <span className="text-xl font-bold tracking-wide">VigilVoice</span>
        </div>
      </div>
      
      <nav className="flex-1 py-6 px-4 space-y-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
                  isActive
                    ? 'bg-primary/20 text-primary font-medium'
                    : 'text-secondary hover:text-white hover:bg-surface-hover'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              {item.label}
            </NavLink>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-border">
        <div className="text-xs text-secondary text-center">
          VigilVoice AI Engine v1.0
        </div>
      </div>
    </aside>
  );
}
