"use client";

import React from 'react';

export type HistoryItem = { id: string; src: string };

export default function History({ items, onSelect }: { items: HistoryItem[]; onSelect: (id: string) => void }) {
	 return (
		 <div className="overflow-y-auto h-[420px] pr-1 space-y-2">
			 {items.map(it => (
				 <div
					 key={it.id}
					 role="button"
					 tabIndex={0}
					 onClick={() => onSelect(it.id)}
					 onKeyDown={(e) => (e.key === 'Enter' ? onSelect(it.id) : null)}
					 className="w-full cursor-pointer select-none focus:outline-none"
				 >
					 <img src={it.src} alt={it.id} className="w-full h-20 object-cover rounded" />
				 </div>
			 ))}
		 </div>
	 );
}
