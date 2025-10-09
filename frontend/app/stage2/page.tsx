"use client";

import React, { useEffect, useRef, useState } from 'react';
import { saveAs } from 'file-saver'; // Add for SVG download (use npm install file-saver if not present)

const BG = "/static/stage2/stage-2-bg.jpg";
const UI = "/static/stage2/stage-2-UI.jpg";

const DESIGN_W = 1280;
const DESIGN_H = 720;

type Rect = { l: number; t: number; w: number; h: number };

export default function Stage2Page() {
	 const [scale, setScale] = useState(1);
	 const [baseW, setBaseW] = useState(DESIGN_W);
	 const [baseH, setBaseH] = useState(DESIGN_H);
	 const [prompt, setPrompt] = useState("");
	 const bgRef = useRef<HTMLImageElement | null>(null);
	 const [debug, setDebug] = useState(false);
	 const [mounted, setMounted] = useState(false);
	 const [image, setImage] = useState<string | null>(null);

	 const [rects, setRects] = useState<Record<string, Rect>>(() => {
		 const def: Record<string, Rect> = {
			 retouch: { l: 31, t: 467, w: 210, h: 95 },
			 resize: { l: 305, t: 466, w: 210, h: 95 },
			 positions: { l: 543, t: 467, w: 210, h: 95 },
			 cancel: { l: 784, t: 468, w: 210, h: 95 },
			 export: { l: 1043, t: 469, w: 210, h: 95 },
			 prompt: { l: 189, t: 604, w: 900, h: 80 },
			 prev: { l: 171, t: 44, w: 64, h: 64 },
			 next: { l: 1042, t: 43, w: 64, h: 64 },
		 };
		 return def;
	 });

	 function recalcScale(nextW: number) {
		 const vw = window.innerWidth;
		 const s = vw / nextW;
		 setScale(s);
	 }

	 useEffect(() => {
		 function onResize() { recalcScale(baseW); }
		 window.addEventListener('resize', onResize);
		 return () => window.removeEventListener('resize', onResize);
	 }, [baseW]);

	 useEffect(() => {
		 try {
			 const params = new URLSearchParams(window.location.search);
			 setDebug(params.get('debug') === '1');
		 } catch {}
	 }, []);

	 useEffect(() => { setMounted(true); }, []);

	 const kx = baseW / DESIGN_W;
	 const ky = baseH / DESIGN_H;
	 const toStyle = (r: Rect) => ({ left: r.l * kx, top: r.t * ky, width: r.w * kx, height: r.h * ky });

	 // Added: Fetch generate image
	 const handleGenerate = async () => {
		 if (!prompt) return;
		 const resp = await fetch('/api/generate', {
			 method: 'POST',
			 headers: { 'Content-Type': 'application/json' },
			 body: JSON.stringify({ prompt })
		 });
		 const data = await resp.json();
		 if (data.image) setImage(data.image); // data:image/png;base64,...
	 };

	 // Wire up actions
	 const handleRetouch = async () => { alert('Retouch action (connect /api/retouch if backend is ready)'); };
	 const handleResize = async () => { alert('Resize action (connect /api/resize if backend is ready)'); };
	 const handlePositions = async () => { alert('Positions action (connect /api/positions if backend is ready)'); };
	 const handleCancel = () => window.location.href = '/';
	 const handleExport = async () => {
		 if (!image) return alert('No image to export');
		 try {
			 const b64 = image.split(',')[1];
			 const resp = await fetch('/api/vectorize', {
				 method: 'POST',
				 headers: { 'Content-Type': 'application/json' },
				 body: JSON.stringify({ image })
			 });
			 if (resp.ok) {
				 const svgText = await resp.text();
				 const blob = new Blob([svgText], { type: 'image/svg+xml' });
				 saveAs(blob, 'vectorized.svg');
			 } else {
				 alert('Vectorization failed');
			 }
		 } catch (err) {
			 alert('Export error: ' + err);
		 }
	 };
	 const handlePrev = () => { try { window.history.back(); } catch {} };
	 const handleNext = () => { try { window.history.forward(); } catch {} };

	 return (
		 <main className="min-h-screen w-full bg-black flex items-start justify-center overflow-auto">
			 <div style={{ width: baseW, height: baseH, transform: `scale(${scale})`, transformOrigin: 'top center' }} className="relative overflow-hidden">
				 <style jsx global>{`
				 .scrollbox{ scrollbar-width: thin; scrollbar-color: #000080 #00000000;}
				 .scrollbox::-webkit-scrollbar{ width: 12px; height: 12px; background: transparent; }
				 .scrollbox::-webkit-scrollbar-track{ background: transparent; }
				 .scrollbox::-webkit-scrollbar-thumb{ background: #000080; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); }
				 .scrollbox::-webkit-scrollbar-thumb:hover{ background: #0000aa; }
				 `}</style>
				 <img
					 ref={bgRef}
					 src={BG}
					 alt=""
					 className="absolute inset-0 w-full h-full object-cover z-0 pointer-events-none"
					 onLoad={(e) => {
						 const img = e.currentTarget;
						 const w = img.naturalWidth || DESIGN_W;
						 const h = img.naturalHeight || DESIGN_H;
						 setBaseW(w);
						 setBaseH(h);
						 recalcScale(w);
					 }}
				 />
				 <img src={UI} alt="" className="absolute inset-0 w-full h-full object-contain z-10 pointer-events-none" />

				 {/* Prev/Next arrows (fixed hotspots) */}
				 {mounted && (
				 <>
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.prev), background: 'transparent', cursor: 'pointer' }} onClick={handlePrev} aria-label="Previous" />
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.next), background: 'transparent', cursor: 'pointer' }} onClick={handleNext} aria-label="Next" />
				 </>
				 )}

				 {/* Prompt (fixed) */}
				 {mounted && (
				 <div className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.prompt), borderRadius: 16 }}>
					 <textarea value={prompt} onChange={e=>setPrompt(e.target.value)} className="w-5/6 h-full bg-transparent outline-none resize-none border-0 scrollbox" style={{ fontSize: '13px', color: '#000080', padding: '6px 10px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere', boxSizing: 'border-box' }} wrap="soft" placeholder="Coloque um avião na imagem..." />
					 <button className="absolute right-2 top-2 bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700" onClick={handleGenerate}>Generate</button>
				 </div>
				 )}
				 {/* Show generated image */}
				 {image && <img src={image} alt="Generated" className="absolute left-1/2 top-20 z-10" style={{ maxWidth: 650, maxHeight: 650, translate: '-50%' }} />}

				 {/* Action buttons (now wired) */}
				 {mounted && (
				 <>
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.retouch), background: 'transparent', cursor: 'pointer' }} onClick={handleRetouch} aria-label="Retocar" />
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.resize), background: 'transparent', cursor: 'pointer' }} onClick={handleResize} aria-label="Redimensionar" />
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.positions), background: 'transparent', cursor: 'pointer' }} onClick={handlePositions} aria-label="Posições" />
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.cancel), background: 'transparent', cursor: 'pointer' }} onClick={handleCancel} aria-label="Cancelar" />
				 	<button className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.export), background: 'transparent', cursor: 'pointer' }} onClick={handleExport} aria-label="Exportar" />
				 </>
				 )}

				 {/* Back to Stage 1 */}
				 <button 
					 className="absolute z-30 bottom-2 left-2 bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition-colors" 
					 onClick={() => window.location.href = '/'}
				 >
					 Stage 1
				 </button>
			 </div>
		 </main>
	 );
}
