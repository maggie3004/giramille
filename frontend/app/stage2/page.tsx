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
	 const [originalImage, setOriginalImage] = useState<string | null>(null);
	 const [retouchPrompt, setRetouchPrompt] = useState("");
	 const [showRetouchModal, setShowRetouchModal] = useState(false);
	 const [showResizeModal, setShowResizeModal] = useState(false);
	 const [resizeWidth, setResizeWidth] = useState(512);
	 const [resizeHeight, setResizeHeight] = useState(512);
  const [imageHistory, setImageHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [imagePosition, setImagePosition] = useState({ x: 0, y: 0 });

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
			 // New: Generate button placed to the right side of the right arrow
			 generate: { l: 1120, t: 52, w: 120, h: 44 },
			 // New: Upload button placed below Generate to avoid overlap
			 upload: { l: 1120, t: 108, w: 140, h: 44 },
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

	 useEffect(() => { 
		 setMounted(true); 
		 // Check if there's a selected image from stage 1
		 const selectedImage = localStorage.getItem('selectedImage');
		 if (selectedImage) {
			 setImage(selectedImage);
			 setOriginalImage(selectedImage);
			 // Add to history
			 setImageHistory([selectedImage]);
			 setHistoryIndex(0);
			 // Clear the stored image
			 localStorage.removeItem('selectedImage');
		 }
	 }, []);

	 const kx = baseW / DESIGN_W;
	 const ky = baseH / DESIGN_H;
	 const toStyle = (r: Rect) => ({ left: r.l * kx, top: r.t * ky, width: r.w * kx, height: r.h * ky });

	 // Image upload handler
	 const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
		 const file = event.target.files?.[0];
		 if (file) {
			 const reader = new FileReader();
			 reader.onload = (e) => {
				 const result = e.target?.result as string;
				 setImage(result);
				 setOriginalImage(result);
				 // Add to history
				 const newHistory = [...imageHistory.slice(0, historyIndex + 1), result];
				 setImageHistory(newHistory);
				 setHistoryIndex(newHistory.length - 1);
			 };
			 reader.readAsDataURL(file);
		 }
	 };

	 // Added: Fetch generate image
	 const handleGenerate = async () => {
		 if (!prompt) return;
		 const resp = await fetch('/api/generate', {
			 method: 'POST',
			 headers: { 'Content-Type': 'application/json' },
			 body: JSON.stringify({ prompt })
		 });
		 const data = await resp.json();
		 if (data.image) {
			 setImage(data.image);
			 setOriginalImage(data.image);
			 // Add to history
			 const newHistory = [...imageHistory.slice(0, historyIndex + 1), data.image];
			 setImageHistory(newHistory);
			 setHistoryIndex(newHistory.length - 1);
		 }
	 };

	 // Wire up actions
	 const handleRetouch = async () => {
		 if (!image) return alert('No image to retouch');
		 setShowRetouchModal(true);
	 };

	 const handleRetouchSubmit = async () => {
		 if (!image || !retouchPrompt.trim()) return;
		 
		 try {
			 const resp = await fetch('/api/retouch', {
				 method: 'POST',
				 headers: { 'Content-Type': 'application/json' },
				 body: JSON.stringify({ 
					 image, 
					 prompt: retouchPrompt 
				 })
			 });
			 const data = await resp.json();
			 if (data.image) {
				 // Add to history
				 const newHistory = [...imageHistory.slice(0, historyIndex + 1), data.image];
				 setImageHistory(newHistory);
				 setHistoryIndex(newHistory.length - 1);
				 setImage(data.image);
				 setShowRetouchModal(false);
				 setRetouchPrompt("");
			 }
		 } catch (error) {
			 alert('Retouch failed: ' + error);
		 }
	 };
	 const handleResize = async () => {
		 if (!image) return alert('No image to resize');
		 setShowResizeModal(true);
	 };

	 const handleResizeSubmit = async () => {
		 if (!image) return;
		 
		 try {
			 const resp = await fetch('/api/resize', {
				 method: 'POST',
				 headers: { 'Content-Type': 'application/json' },
				 body: JSON.stringify({ 
					 image, 
					 width: resizeWidth, 
					 height: resizeHeight 
				 })
			 });
			 const data = await resp.json();
			 if (data.image) {
				 // Add to history
				 const newHistory = [...imageHistory.slice(0, historyIndex + 1), data.image];
				 setImageHistory(newHistory);
				 setHistoryIndex(newHistory.length - 1);
				 setImage(data.image);
				 setShowResizeModal(false);
			 }
		 } catch (error) {
			 alert('Resize failed: ' + error);
		 }
	 };
	 const handlePositions = async () => {
		 if (!image) return alert('No image for positions');
		 const resp = await fetch('/api/positions', {
			 method: 'POST',
			 headers: { 'Content-Type': 'application/json' },
			 body: JSON.stringify({ image })
		 });
		 const data = await resp.json();
		 if (data.image) setImage(data.image);
	 };
	 const handleCancel = () => {
		 // Cancel all changes - reset to original image
		 if (originalImage) {
			 setImage(originalImage);
			 // Reset history to only contain the original image
			 setImageHistory([originalImage]);
			 setHistoryIndex(0);
		 }
	 };
	 const handleExport = async () => {
		 if (!image) return alert('No image to export');
		 try {
			 // Convert base64 to blob and download
			 const b64 = image.split(',')[1];
			 const byteCharacters = atob(b64);
			 const byteNumbers = new Array(byteCharacters.length);
			 for (let i = 0; i < byteCharacters.length; i++) {
				 byteNumbers[i] = byteCharacters.charCodeAt(i);
			 }
			 const byteArray = new Uint8Array(byteNumbers);
			 const blob = new Blob([byteArray], { type: 'image/png' });
			 saveAs(blob, 'edited-image.png');
		 } catch (err) {
			 alert('Export error: ' + err);
		 }
	 };

	 // Zoom and pan handlers
	 const handleZoomIn = () => {
		 setZoomLevel(prev => Math.min(prev * 1.2, 5));
	 };

	 const handleZoomOut = () => {
		 setZoomLevel(prev => Math.max(prev / 1.2, 0.1));
	 };

	 const handleResetZoom = () => {
		 setZoomLevel(1);
		 setImagePosition({ x: 0, y: 0 });
	 };

	 const handleMouseDown = (e: React.MouseEvent) => {
		 setIsDragging(true);
		 setDragStart({ x: e.clientX - imagePosition.x, y: e.clientY - imagePosition.y });
	 };

	 const handleMouseMove = (e: React.MouseEvent) => {
		 if (isDragging) {
			 setImagePosition({
				 x: e.clientX - dragStart.x,
				 y: e.clientY - dragStart.y
			 });
		 }
	 };

	 const handleMouseUp = () => {
		 setIsDragging(false);
	 };

	 const handleWheel = (e: React.WheelEvent) => {
		 e.preventDefault();
		 const delta = e.deltaY > 0 ? 0.9 : 1.1;
		 setZoomLevel(prev => Math.max(0.1, Math.min(5, prev * delta)));
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
	 	{/* Generate near right arrow */}
	 	<button
	 		 className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''} bg-blue-600 text-white rounded shadow hover:bg-blue-700`}
	 		 style={{ ...toStyle(rects.generate) }}
	 		 onClick={handleGenerate}
	 	>
	 		Generate
	 	</button>
				 </>
				 )}

				 {/* Prompt (fixed) */}
				 {mounted && (
				 <div className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.prompt), borderRadius: 16 }}>
					 <textarea value={prompt} onChange={e=>setPrompt(e.target.value)} className="w-5/6 h-full bg-transparent outline-none resize-none border-0 scrollbox" style={{ fontSize: '13px', color: '#000080', padding: '6px 10px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere', boxSizing: 'border-box' }} wrap="soft" placeholder="Coloque um avião na imagem..." />
				 </div>
				 )}
	 {/* Image Upload Button (aligned below Generate via rects.upload) */}
	 <div className={`absolute z-20 ${debug ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.upload) }}>
					 <input
						 type="file"
						 accept="image/*"
						 onChange={handleImageUpload}
						 className="hidden"
						 id="image-upload"
					 />
					 <label
						 htmlFor="image-upload"
			 className="bg-blue-600 text-white w-full h-full px-4 py-2 rounded cursor-pointer hover:bg-blue-700 transition-colors flex items-center justify-center"
					 >
						 Upload Image
					 </label>
				 </div>

				 {/* Show generated/uploaded image with zoom and pan */}
				 {image && (
					 <div 
						 className="absolute left-1/2 top-20 z-10 overflow-hidden"
						 style={{ 
							 translate: '-50%', 
							 width: '450px', 
							 height: '300px',
							 border: '2px solid rgba(255,255,255,0.3)',
							 borderRadius: '8px',
							 backgroundColor: 'rgba(0,0,0,0.1)'
						 }}
					 >
						 <img 
							 src={image} 
							 alt="Generated" 
							 className="w-full h-full object-contain cursor-move select-none"
							 style={{ 
								 transform: `scale(${zoomLevel}) translate(${imagePosition.x / zoomLevel}px, ${imagePosition.y / zoomLevel}px)`,
								 transformOrigin: 'center center',
								 transition: isDragging ? 'none' : 'transform 0.1s ease-out'
							 }}
							 onMouseDown={handleMouseDown}
							 onMouseMove={handleMouseMove}
							 onMouseUp={handleMouseUp}
							 onMouseLeave={handleMouseUp}
							 onWheel={handleWheel}
							 draggable={false}
						 />
					 </div>
				 )}

	 {/* Zoom Controls */}
	 {image && (
		 <div className="absolute left-1/2 top-10 z-20 flex gap-2" style={{ translate: '-50%' }}>
			 <button
				 onClick={handleZoomOut}
				 className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 text-sm"
				 title="Zoom Out"
			 >
				 -
			 </button>
			 <span className="bg-blue-600 text-white px-3 py-1 rounded text-sm">
				 {Math.round(zoomLevel * 100)}%
			 </span>
			 <button
				 onClick={handleZoomIn}
				 className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 text-sm"
				 title="Zoom In"
			 >
				 +
			 </button>
			 <button
				 onClick={handleResetZoom}
				 className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 text-sm"
				 title="Reset Zoom"
			 >
				 Reset
			 </button>
		 </div>
	 )}

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

				 {/* Retouch Modal */}
				 {showRetouchModal && (
					 <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
						 <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
							 <h3 className="text-lg font-semibold mb-4 text-gray-800">Retouch Image</h3>
							 <textarea
								 value={retouchPrompt}
								 onChange={(e) => setRetouchPrompt(e.target.value)}
								 placeholder="Describe what changes you want to make to the image..."
								 className="w-full h-24 p-3 border border-gray-300 rounded mb-4 resize-none text-gray-800 placeholder-gray-500"
							 />
							 <div className="flex gap-2 justify-end">
								 <button
									 onClick={() => setShowRetouchModal(false)}
									 className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
								 >
									 Cancel
								 </button>
								 <button
									 onClick={handleRetouchSubmit}
									 className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
								 >
									 Apply Changes
								 </button>
							 </div>
						 </div>
					 </div>
				 )}

				 {/* Resize Modal */}
				 {showResizeModal && (
					 <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
						 <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
							 <h3 className="text-lg font-semibold mb-4 text-gray-800">Resize Image</h3>
							 <div className="space-y-4">
								 <div>
									 <label className="block text-sm font-medium mb-2 text-gray-700">Width (pixels)</label>
									 <input
										 type="number"
										 value={resizeWidth}
										 onChange={(e) => setResizeWidth(parseInt(e.target.value) || 512)}
										 className="w-full p-2 border border-gray-300 rounded text-gray-800"
										 min="1"
										 max="2048"
									 />
								 </div>
								 <div>
									 <label className="block text-sm font-medium mb-2 text-gray-700">Height (pixels)</label>
									 <input
										 type="number"
										 value={resizeHeight}
										 onChange={(e) => setResizeHeight(parseInt(e.target.value) || 512)}
										 className="w-full p-2 border border-gray-300 rounded text-gray-800"
										 min="1"
										 max="2048"
									 />
								 </div>
							 </div>
							 <div className="flex gap-2 justify-end mt-6">
								 <button
									 onClick={() => setShowResizeModal(false)}
									 className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
								 >
									 Cancel
								 </button>
								 <button
									 onClick={handleResizeSubmit}
									 className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
								 >
									 Resize Image
								 </button>
							 </div>
						 </div>
					 </div>
				 )}
			 </div>
		 </main>
	 );
}
