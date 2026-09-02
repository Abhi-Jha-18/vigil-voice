import React from 'react';

export function PageContainer({ title, children }) {
  return (
    <main className="flex-1 overflow-y-auto p-6 md:p-8 w-full max-w-7xl mx-auto">
      {title && (
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-white">{title}</h1>
        </div>
      )}
      <div className="w-full">
        {children}
      </div>
    </main>
  );
}
