"use client";

import React, { MouseEvent, useEffect, useRef, useState } from 'react';

const BG = "/static/stage1/stage-1-bg.jpg";
const UI = "/static/stage1/stage-1-UI.png";

const DESIGN_W = 1280;
const DESIGN_H = 720;

type Rect = { l: number; t: number; w: number; h: number };

export default function HomePage() {
	 const [scale, setScale] = useState(1);
	 const [baseW, setBaseW] = useState(DESIGN_W);
	 const [baseH, setBaseH] = useState(DESIGN_H);
	 const [prompt, setPrompt] = useState("");
	 const [historyImgs, setHistoryImgs] = useState<string[]>([
		 // Add some test images to make scrollbar visible
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#ff6b6b"/><text x="20" y="48" font-size="20" fill="#fff">Test 1</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#4ecdc4"/><text x="20" y="48" font-size="20" fill="#fff">Test 2</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#45b7d1"/><text x="20" y="48" font-size="20" fill="#fff">Test 3</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#96ceb4"/><text x="20" y="48" font-size="20" fill="#fff">Test 4</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#feca57"/><text x="20" y="48" font-size="20" fill="#fff">Test 5</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#ff9ff3"/><text x="20" y="48" font-size="20" fill="#fff">Test 6</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#54a0ff"/><text x="20" y="48" font-size="20" fill="#fff">Test 7</text></svg>`),
		 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#5f27cd"/><text x="20" y="48" font-size="20" fill="#fff">Test 8</text></svg>`),
	 ]);
	 const [preview, setPreview] = useState<string | null>(null);
	 const [calib, setCalib] = useState(false);
	 const [drag, setDrag] = useState<{ key: string; dx: number; dy: number } | null>(null);
	 const bgRef = useRef<HTMLImageElement | null>(null);

	 const [rects, setRects] = useState<Record<string, Rect>>(() => {
		 const def: Record<string, Rect> = {
			 prompt: { l: 60, t: 180, w: 350, h: 270 },
			 vecBtn: { l: 510, t: 325, w: 280, h: 105 },
			 pngBtn: { l: 510, t: 490, w: 280, h: 105 },
			 hist: { l: 935, t: 192, w: 320, h: 398 },
		 };
		 try {
			 const saved = localStorage.getItem("ui_rects");
			 if (saved) return { ...def, ...JSON.parse(saved) };
		 } catch {}
		 return def;
	 });

	 function saveRects(next: Record<string, Rect>) {
		 setRects(next);
		 try { localStorage.setItem("ui_rects", JSON.stringify(next)); } catch {}
	 }

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

	 const kx = baseW / DESIGN_W;
	 const ky = baseH / DESIGN_H;
	 const toStyle = (r: Rect) => ({ left: r.l * kx, top: r.t * ky, width: r.w * kx, height: r.h * ky });

	 const startDrag = (e: MouseEvent<HTMLDivElement | HTMLButtonElement>, key: string) => {
		 if (!calib) return;
		 e.preventDefault();
		 setDrag({ key, dx: e.clientX, dy: e.clientY });
	 };
	 const onMove = (e: MouseEvent<HTMLDivElement>) => {
		 if (!drag) return;
		 const { key, dx, dy } = drag;
		 const dpx = (e.clientX - dx) / kx;
		 const dpy = (e.clientY - dy) / ky;
		 const cur = rects[key];
		 const next = { ...rects, [key]: { ...cur, l: Math.max(0, cur.l + dpx), t: Math.max(0, cur.t + dpy) } };
		 setDrag({ key, dx: e.clientX, dy: e.clientY });
		 saveRects(next);
	 };
	 const endDrag = () => setDrag(null);

	 // Dataset images mapping for Giramille style
	 const datasetImages = {
		 // Animals
		 'bird': ['Passaros.png', 'borboleta cozinhando.png', 'borboletinha.png', 'borbolitinha.png'],
		 'cat': ['Cat.jpg', 'Cat.png'],
		 'dog': ['Dog.jpg', 'Dog.png', 'Dog-Femea.png', 'Dog-menina.png', 'Dog-menino.png', 'dog-correndo.png'],
		 'fish': ['peixe.png', 'salmao.png', 'pirarucu1.png'],
		 'butterfly': ['borboleta cozinhando.png', 'borboletinha.png', 'borbolitinha.png'],
		 'horse': ['Horse.jpg', 'Cavalinho.png'],
		 'bear': ['Bear.jpg'],
		 'chick': ['Chick.jpg', 'Chick2.jpg', 'PINTINHO_1.png', 'pintinho_2.png', 'pintinho.png'],
		 'ant': ['Ant.jpg'],
		 'frog': ['sapo-pirata.png', 'sapo.png'],
		 'crocodile': ['Jacaré.png'],
		 'moose': ['alce.jpg'],
		 't-rex': ['T-Rex.jpg'],
		 'fairy': ['Fairy.jpg'],
		 'witch': ['Witch.jpg'],
		 
		 // Objects
		 'car': ['Carros.jpg', 'Carrinho.png', 'Jeep 3-4.png', 'Jeep poses.png'],
		 'airplane': ['aviao.jpg'],
		 'train': ['Train.jpg', 'train.png', 'Train Lado.png', 'Train Virando.png', 'frente trem.png'],
		 'bus': ['Bus.jpg', 'Onibus.jpg'],
		 'boat': ['Barcos.jpg', 'boat-trail.png'],
		 'house': ['Casa da Giramille.png', 'Casa Giramille.png', 'Casa-Giramille.png', 'Casa-dentro.png'],
		 'castle': ['Castelo_final 01.png', 'Castelo-Salao-Nobre.png'],
		 'tree': ['Maple tree.png', 'floresta [Converted].png', 'Floresta.png', 'clipart-for-tree-11.png'],
		 'flower': ['Margarida.png', 'Campo de flores.png', 'Campo de flores Colorido.png'],
		 'hat': ['chape.png', 'chapeu.png', 'sombrero M.png', 'hat-icon-5277590_1280.png'],
		 'star': ['estrela.jpg', 'Estrelinha.png', 'Estrelinha pt 2.png', '10df7ee30a45905aca812b9d082366d8-oito-pontas-estrela-marrom.png'],
		 'heart': ['Coracao.png', 'Coracao-png.png', 'Corações PNG-01.png', 'coração girafa.png'],
		 'sun': ['Sol Brilhar-01.png', 'Nuvens e Sol.jpg'],
		 'moon': ['lua grande.png'],
		 'cloud': ['Nuvens e Sol.jpg'],
		 'ball': ['popsicle.png', 'picole.png'],
		 'book': ['Livrinho.png', 'Livro1.png'],
		 'cup': ['caneca cor1.png', 'caneca cor2.png', 'caneca cor3.png', 'copo GiramilleC.png'],
		 'crown': ['coroa.png'],
		 'key': ['Chave.png'],
		 'coin': ['Moedas-01-01.png'],
		 'guitar': ['Violão.png'],
		 'phone': ['celular.png'],
		 'shoes': ['Sapato - Loira.png', 'salto alto brilhante.png'],
		 'sofa': ['sofa.png'],
		 'bed': ['Cama da Giramille -2.png', 'cama.png', 'cama_1.png'],
		 'table': ['Mesa Parabens.png'],
		 'chair': ['Suporte do Guga.png', 'Suporte Guga.png'],
		 'lamp': ['luz.png'],
		 'tv': ['Icone_tv.jpg'],
		 'music': ['music.png'],
		 'game': ['games.png'],
		 'social': ['social.png'],
		 
		 // Scenes/Backgrounds
		 'forest': ['floresta [Converted].png', 'Floresta.png', 'fundo mata.png', 'forest.jpg'],
		 'beach': ['praia.png', 'Fundo-praia-vinheta.png', 'mar2.png'],
		 'mountain': ['montanha.png', 'montanha_1.png'],
		 'city': ['Nova York - Paisagem.jpg', 'Empire State.jpg', 'Times Square.jpg', 'Central Park.jpg', 'Museu de Arte de Nova York.jpg', 'Rockefeller Center.jpg', 'Ponte Brooklin.jpg'],
		 'school': ['escola.jpg', 'Sala de aula.png', 'Sala Recreativa.png', 'Lousa-Video-Aula.png', 'lousa.png'],
		 'farm': ['fazenda.png', 'Cenário Celeiro.png'],
		 'prison': ['Prisão-01.png'],
		 'stage': ['palco.png', 'Ref - Show.jpg', 'show.png'],
		 'park': ['Central Park.jpg'],
		 'bridge': ['Ponte Brooklin.jpg'],
		 'statue': ['Estátua da Liberdade (ela fica numa ilha).jpg'],
		 'sky': ['Céu.png', 'Ceu surgindo.png'],
		 'ground': ['chão.jpg', 'grass.jpg'],
		 'wood': ['madeira.png'],
		 'water': ['Traco na agua.png'],
		 'rail': ['trilho-s.png', 'trilho2.png'],
		 
		 // Food
		 'apple': ['apple.png'],
		 'bread': ['pao.png'],
		 'milk': ['Leite.png', 'leite em po.png'],
		 'banana': ['Bananinhas.png'],
		 'ice cream': ['popsicle.png', 'picole.png'],
		 'fish food': ['Racao cats.png', 'Racao dogs.png', 'Dog-Food-02.png'],
		 
		 // Characters
		 'giramille': ['Pai-Francisco.png', 'Girafa_corpo.png'],
		 'indian': ['Indios.png', 'Indios2.png', 'India1.png', 'India2.png'],
		 'firefighter': ['Bombeiro.jpg'],
		 'chef': ['borboleta cozinhando.png', 'borboleta cozinhando_1.png'],
		 
		 // Items
		 'wand': ['VARINHA-GIRAFA.png'],
		 'fishing rod': ['vara_pescar.png'],
		 'surfboard': ['Surfboard_clip_art_hight.png'],
		 'mask': ['kisspng-zorro-dominoes-domino-mask-computer-icons-mask-black-5b3093633591a8.1803331115299101152194.png'],
		 'flag': ['bandeira-do-quadrado-da-textura-do-giz-no-quadro-negro-50077734.jpg'],
		 'map': ['Mapa Mundi.png'],
		 'leaf': ['folha.png', 'folha canada.png'],
		 'rainbow': ['rainbow-clipart-84-700x409.png', '9436a75cdabae1ea6e7cacb9f3bb952d-desenho-colorido-arco-iris.png'],
		 'clothespin': ['prendedor.png'],
		 'belt': ['Cinturao_Prancheta 1.png'],
		 'tutu': ['tutu_Prancheta 1.png'],
		 'bow': ['Laço Giramille.jpg', 'lanco.png'],
		 'frame': ['Moldura retangular2.png'],
		 'sign': ['Placa_madeira.png', 'Placa_madeira_logo.png', 'Plaquinha_Vetor.png'],
		 
		 // Colors
		 'red': ['vermelho.png', 'Coracao.png'],
		 'blue': ['azul.png'],
		 'green': ['verde.png'],
		 'yellow': ['amarelo.png'],
		 'purple': ['roxo.png'],
		 'pink': ['rosa.png'],
		 'brown': ['marrom.png'],
		 'black': ['preto.png'],
		 'white': ['branco.png'],
		 'orange': ['laranja.png'],
		 
		 // Holidays/Events
		 'christmas': ['Natal-2.png', 'christmas-background-with-pastel-bokeh-lights-stars-design.jpg'],
		 'easter': ['Páscoa.png'],
		 'birthday': ['aniversário.png', 'bg_aniver Página 2.png'],
		 'congratulations': ['Parabéns.png'],
		 
		 // Hygiene/Products
		 'shampoo': ['shampoo.png', 'shampoo-2.png', 'shampoo 1.png', 'shampoo2.png'],
		 'soap': ['sabonete.png'],
		 'toothbrush': ['pasta escova de dente.png'],
		 'mouthwash': ['Enxaguante bucal.png'],
		 'dental floss': ['fio dental.png'],
		 'diaper': ['fraldas.png'],
		 'diaper cream': ['Pomada para assaduras.png'],
		 'conditioner': ['condicionador.png'],
		 'wet wipes': ['lenço umedecido.png'],
		 'hand sanitizer': ['gel antisseptico.png']
	 };

	const generateImageFromPrompt = async (prompt: string, type: 'png' | 'vector'): Promise<string> => {
		console.log('AI Image Generation - Prompt:', prompt, 'Type:', type);

		try {
			console.log('🔄 Calling backend API...');
			// Call backend API for AI generation
			const response = await fetch('http://localhost:5000/generate', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					prompt: prompt,
					style: type === 'vector' ? 'cartoon' : 'cartoon',
					quality: 'balanced',
					width: 512,
					height: 512
				})
			});
			console.log('📡 API Response status:', response.status);

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			const data = await response.json();
			
			if (data.success && data.image) {
				console.log('✅ AI Generated image from backend:', data.image.length);
				return data.image;
			} else {
				throw new Error('No image generated');
			}
		} catch (error) {
			console.error('❌ Backend API error:', error);
			
			// Fallback to local generation
			console.log('🔄 Using fallback local generation...');
			const canvas = document.createElement('canvas');
			canvas.width = 512;
			canvas.height = 512;
			const ctx = canvas.getContext('2d')!;
			ctx.clearRect(0, 0, 512, 512);
			generateAIArt(ctx, prompt, type);
			return canvas.toDataURL('image/png');
		}
	};

	 const generateAIArt = (ctx: CanvasRenderingContext2D, prompt: string, type: 'png' | 'vector') => {
		 const lowerPrompt = prompt.toLowerCase();
		 
		 // AI Style Analysis
		 const isRealistic = lowerPrompt.includes('realistic') || lowerPrompt.includes('photo') || lowerPrompt.includes('photograph');
		 const isCartoon = lowerPrompt.includes('cartoon') || lowerPrompt.includes('anime') || lowerPrompt.includes('manga');
		 const isAbstract = lowerPrompt.includes('abstract') || lowerPrompt.includes('artistic') || lowerPrompt.includes('creative');
		 const isMinimalist = lowerPrompt.includes('minimal') || lowerPrompt.includes('simple') || lowerPrompt.includes('clean');
		 
		 // Color Analysis
		 const colors = extractColorsFromPrompt(prompt);
		 const primaryColor = colors.primary;
		 const secondaryColor = colors.secondary;
		 const accentColor = colors.accent;
		 
		 // Object Detection
		 const objects = detectObjectsInPrompt(prompt);
		 
		 if (type === 'png') {
			 generatePNGStyle(ctx, prompt, objects, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon, isAbstract, isMinimalist);
		 } else {
			 generateVectorStyle(ctx, prompt, objects, primaryColor, secondaryColor, accentColor, isMinimalist);
		 }
	 };

	 const extractColorsFromPrompt = (prompt: string) => {
		 const lowerPrompt = prompt.toLowerCase();
		 const colorMap: { [key: string]: string } = {
			 'red': '#ff4757', 'blue': '#3742fa', 'green': '#2ed573', 'yellow': '#ffa502',
			 'purple': '#5f27cd', 'orange': '#ff6348', 'pink': '#ff3838', 'brown': '#8b4513',
			 'black': '#2c2c54', 'white': '#f8f9fa', 'gray': '#57606f', 'cyan': '#0abde3',
			 'magenta': '#ff6b6b', 'lime': '#32ff7e', 'indigo': '#4834d4', 'violet': '#9c88ff'
		 };
		 
		 const foundColors: string[] = [];
		 for (const [colorName, colorValue] of Object.entries(colorMap)) {
			 if (lowerPrompt.includes(colorName)) {
				 foundColors.push(colorValue);
			 }
		 }
		 
		 return {
			 primary: foundColors[0] || '#4ecdc4',
			 secondary: foundColors[1] || '#ff6b6b',
			 accent: foundColors[2] || '#feca57'
		 };
	 };

	 const detectObjectsInPrompt = (prompt: string) => {
		 const lowerPrompt = prompt.toLowerCase();
		 const objectKeywords = {
			 'house': ['house', 'home', 'building', 'casa', 'casa'],
			 'car': ['car', 'vehicle', 'auto', 'carro'],
			 'tree': ['tree', 'plant', 'forest', 'árvore'],
			 'person': ['person', 'people', 'man', 'woman', 'child', 'pessoa'],
			 'animal': ['cat', 'dog', 'bird', 'fish', 'animal', 'gato', 'cachorro'],
			 'nature': ['mountain', 'ocean', 'sky', 'cloud', 'sun', 'moon', 'montanha'],
			 'food': ['food', 'fruit', 'apple', 'banana', 'comida'],
			 'abstract': ['abstract', 'pattern', 'design', 'art', 'arte']
		 };
		 
		 const detectedObjects: string[] = [];
		 for (const [category, keywords] of Object.entries(objectKeywords)) {
			 if (keywords.some(keyword => lowerPrompt.includes(keyword))) {
				 detectedObjects.push(category);
			 }
		 }
		 
		 return detectedObjects.length > 0 ? detectedObjects : ['abstract'];
	 };

	 const generatePNGStyle = (ctx: CanvasRenderingContext2D, prompt: string, objects: string[], primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean, isAbstract: boolean, isMinimalist: boolean) => {
		 // AI-Generated PNG Style Art
		 
		 // Background
		 if (isAbstract) {
			 // Abstract background
			 const gradient = ctx.createRadialGradient(256, 256, 0, 256, 256, 300);
			 gradient.addColorStop(0, primaryColor + '40');
			 gradient.addColorStop(0.5, secondaryColor + '20');
			 gradient.addColorStop(1, accentColor + '10');
			 ctx.fillStyle = gradient;
			 ctx.fillRect(0, 0, 512, 512);
		 } else {
			 // Realistic background
			 const gradient = ctx.createLinearGradient(0, 0, 512, 512);
			 gradient.addColorStop(0, primaryColor + '30');
			 gradient.addColorStop(1, secondaryColor + '20');
			 ctx.fillStyle = gradient;
			 ctx.fillRect(0, 0, 512, 512);
		 }
		 
		 // Add AI-generated elements based on detected objects
		 objects.forEach((obj, index) => {
			 drawAIObject(ctx, obj, primaryColor, secondaryColor, accentColor, index, isRealistic, isCartoon, isAbstract);
		 });
		 
		 // Add AI-generated details
		 addAIDetails(ctx, prompt, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon, isAbstract);
		 
		 // Add prompt as watermark
		 ctx.fillStyle = 'rgba(0,0,0,0.3)';
		 ctx.font = 'bold 16px Arial';
		 ctx.textAlign = 'center';
		 ctx.fillText(prompt, 256, 480);
	 };

	 const generateVectorStyle = (ctx: CanvasRenderingContext2D, prompt: string, objects: string[], primaryColor: string, secondaryColor: string, accentColor: string, isMinimalist: boolean) => {
		 // AI-Generated Vector Style Art
		 
		 // Clean background
		 ctx.fillStyle = '#ffffff';
		 ctx.fillRect(0, 0, 512, 512);
		 
		 // Vector elements
		 objects.forEach((obj, index) => {
			 drawVectorObject(ctx, obj, primaryColor, secondaryColor, accentColor, index, isMinimalist);
		 });
		 
		 // Add vector details
		 addVectorDetails(ctx, prompt, primaryColor, secondaryColor, accentColor, isMinimalist);
		 
		 // Add prompt as watermark
		 ctx.fillStyle = 'rgba(0,0,0,0.5)';
		 ctx.font = 'bold 14px Arial';
		 ctx.textAlign = 'center';
		 ctx.fillText(prompt, 256, 480);
	 };

	 const drawAIObject = (ctx: CanvasRenderingContext2D, object: string, primaryColor: string, secondaryColor: string, accentColor: string, index: number, isRealistic: boolean, isCartoon: boolean, isAbstract: boolean) => {
		 const x = 100 + (index * 150) + Math.random() * 100;
		 const y = 150 + Math.random() * 200;
		 const size = 60 + Math.random() * 80;
		 
		 ctx.fillStyle = primaryColor;
		 ctx.strokeStyle = secondaryColor;
		 ctx.lineWidth = isRealistic ? 2 : 4;
		 
		 switch (object) {
			 case 'house':
				 drawAIHouse(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 case 'car':
				 drawAICar(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 case 'tree':
				 drawAITree(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 case 'person':
				 drawAIPerson(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 case 'animal':
				 drawAIAnimal(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 case 'nature':
				 drawAINature(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 case 'food':
				 drawAIFood(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isRealistic, isCartoon);
				 break;
			 default:
				 drawAIAbstract(ctx, x, y, size, primaryColor, secondaryColor, accentColor, isAbstract);
		 }
	 };

	 const drawAIHouse = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated House
		 ctx.fillStyle = primaryColor;
		 
		 // House base
		 ctx.fillRect(x - size/2, y - size/3, size, size * 0.6);
		 
		 // Roof
		 ctx.fillStyle = secondaryColor;
		 ctx.beginPath();
		 ctx.moveTo(x - size/2, y - size/3);
		 ctx.lineTo(x, y - size/2);
		 ctx.lineTo(x + size/2, y - size/3);
		 ctx.closePath();
		 ctx.fill();
		 
		 // Door
		 ctx.fillStyle = accentColor;
		 ctx.fillRect(x - size/8, y - size/6, size/4, size/3);
		 
		 // Windows
		 ctx.fillStyle = '#87ceeb';
		 ctx.fillRect(x - size/3, y - size/4, size/6, size/6);
		 ctx.fillRect(x + size/6, y - size/4, size/6, size/6);
		 
		 // Add AI details
		 if (isRealistic) {
			 // Add shadows and highlights
			 ctx.fillStyle = 'rgba(0,0,0,0.2)';
			 ctx.fillRect(x - size/2, y + size/6, size, 4);
		 }
	 };

	 const drawAICar = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated Car
		 ctx.fillStyle = primaryColor;
		 
		 // Car body
		 ctx.fillRect(x - size/2, y - size/4, size, size/2);
		 
		 // Wheels
		 ctx.fillStyle = '#2c2c54';
		 ctx.beginPath();
		 ctx.arc(x - size/3, y + size/4, size/8, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + size/3, y + size/4, size/8, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Windows
		 ctx.fillStyle = '#87ceeb';
		 ctx.fillRect(x - size/3, y - size/6, size/3, size/6);
	 };

	 const drawAITree = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated Tree
		 // Trunk
		 ctx.fillStyle = '#8b4513';
		 ctx.fillRect(x - size/12, y - size/6, size/6, size/3);
		 
		 // Leaves
		 ctx.fillStyle = primaryColor;
		 ctx.beginPath();
		 ctx.arc(x, y - size/3, size/3, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawAIPerson = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated Person
		 // Head
		 ctx.fillStyle = primaryColor;
		 ctx.beginPath();
		 ctx.arc(x, y - size/3, size/6, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Body
		 ctx.fillStyle = secondaryColor;
		 ctx.fillRect(x - size/8, y - size/6, size/4, size/3);
		 
		 // Arms
		 ctx.fillRect(x - size/4, y - size/8, size/6, size/8);
		 ctx.fillRect(x + size/12, y - size/8, size/6, size/8);
		 
		 // Legs
		 ctx.fillRect(x - size/12, y + size/8, size/12, size/4);
		 ctx.fillRect(x, y + size/8, size/12, size/4);
	 };

	 const drawAIAnimal = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated Animal (generic)
		 ctx.fillStyle = primaryColor;
		 
		 // Body
		 ctx.beginPath();
		 ctx.ellipse(x, y, size/3, size/4, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Head
		 ctx.beginPath();
		 ctx.arc(x, y - size/4, size/6, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Ears
		 ctx.fillStyle = secondaryColor;
		 ctx.beginPath();
		 ctx.arc(x - size/8, y - size/3, size/12, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + size/8, y - size/3, size/12, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawAINature = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated Nature element
		 ctx.fillStyle = primaryColor;
		 
		 // Mountain
		 ctx.beginPath();
		 ctx.moveTo(x - size/2, y + size/4);
		 ctx.lineTo(x, y - size/4);
		 ctx.lineTo(x + size/2, y + size/4);
		 ctx.closePath();
		 ctx.fill();
		 
		 // Sun
		 ctx.fillStyle = accentColor;
		 ctx.beginPath();
		 ctx.arc(x + size/3, y - size/3, size/8, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawAIFood = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean) => {
		 // AI-Generated Food
		 ctx.fillStyle = primaryColor;
		 
		 // Apple
		 ctx.beginPath();
		 ctx.arc(x, y, size/4, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Stem
		 ctx.fillStyle = secondaryColor;
		 ctx.fillRect(x - size/16, y - size/4, size/8, size/8);
	 };

	 const drawAIAbstract = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, primaryColor: string, secondaryColor: string, accentColor: string, isAbstract: boolean) => {
		 // AI-Generated Abstract Art
		 ctx.fillStyle = primaryColor;
		 
		 // Abstract shapes
		 for (let i = 0; i < 5; i++) {
			 const angle = (i * Math.PI * 2) / 5;
			 const shapeX = x + Math.cos(angle) * size/3;
			 const shapeY = y + Math.sin(angle) * size/3;
			 
			 ctx.beginPath();
			 ctx.arc(shapeX, shapeY, size/8, 0, 2 * Math.PI);
			 ctx.fill();
		 }
	 };

	 const drawVectorObject = (ctx: CanvasRenderingContext2D, object: string, primaryColor: string, secondaryColor: string, accentColor: string, index: number, isMinimalist: boolean) => {
		 const x = 100 + (index * 150) + Math.random() * 100;
		 const y = 150 + Math.random() * 200;
		 const size = 40 + Math.random() * 60;
		 
		 ctx.fillStyle = primaryColor;
		 ctx.strokeStyle = secondaryColor;
		 ctx.lineWidth = 3;
		 
		 // Simplified vector versions
		 switch (object) {
			 case 'house':
				 // Simple vector house
				 ctx.fillRect(x - size/2, y - size/3, size, size * 0.6);
				 ctx.beginPath();
				 ctx.moveTo(x - size/2, y - size/3);
				 ctx.lineTo(x, y - size/2);
				 ctx.lineTo(x + size/2, y - size/3);
				 ctx.closePath();
				 ctx.fill();
				 break;
			 case 'car':
				 // Simple vector car
				 ctx.fillRect(x - size/2, y - size/4, size, size/2);
				 break;
			 default:
				 // Generic vector shape
				 ctx.beginPath();
				 ctx.arc(x, y, size/2, 0, 2 * Math.PI);
				 ctx.fill();
		 }
	 };

	 const addAIDetails = (ctx: CanvasRenderingContext2D, prompt: string, primaryColor: string, secondaryColor: string, accentColor: string, isRealistic: boolean, isCartoon: boolean, isAbstract: boolean) => {
		 // Add AI-generated atmospheric details
		 if (isRealistic) {
			 // Add realistic details
			 ctx.fillStyle = 'rgba(255,255,255,0.1)';
			 for (let i = 0; i < 20; i++) {
				 ctx.fillRect(Math.random() * 512, Math.random() * 512, 2, 2);
			 }
		 } else if (isCartoon) {
			 // Add cartoon details
			 ctx.fillStyle = accentColor;
			 for (let i = 0; i < 10; i++) {
				 ctx.beginPath();
				 ctx.arc(Math.random() * 512, Math.random() * 512, 3, 0, 2 * Math.PI);
				 ctx.fill();
			 }
		 }
	 };

	 const addVectorDetails = (ctx: CanvasRenderingContext2D, prompt: string, primaryColor: string, secondaryColor: string, accentColor: string, isMinimalist: boolean) => {
		 // Add vector details
		 ctx.strokeStyle = accentColor;
		 ctx.lineWidth = 2;
		 
		 // Add geometric lines
		 for (let i = 0; i < 5; i++) {
			 ctx.beginPath();
			 ctx.moveTo(Math.random() * 512, Math.random() * 512);
			 ctx.lineTo(Math.random() * 512, Math.random() * 512);
			 ctx.stroke();
		 }
	 };

	 // Legacy function - now replaced by AI generation system
	 const generateDynamicContent = (ctx: CanvasRenderingContext2D, color: string, prompt: string, type: 'png' | 'vector') => {
		 // This function is now deprecated - using AI generation instead
		 generateAIArt(ctx, prompt, type);
	 };

	 // Legacy functions removed - using new AI generation system instead

	 // Drawing functions for different objects
	 const drawHat = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Hat crown
		 ctx.beginPath();
		 ctx.ellipse(x, y - 10, 25, 15, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Hat brim
		 ctx.beginPath();
		 ctx.ellipse(x, y + 5, 35, 8, 0, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorHat = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple hat
		 ctx.beginPath();
		 ctx.ellipse(x, y - 15, 30, 20, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x, y + 10, 40, 10, 0, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawMountain = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Mountain peaks
		 ctx.beginPath();
		 ctx.moveTo(x - 40, y + 20);
		 ctx.lineTo(x - 20, y - 20);
		 ctx.lineTo(x, y + 10);
		 ctx.lineTo(x + 20, y - 10);
		 ctx.lineTo(x + 40, y + 20);
		 ctx.fill();
	 };

	 const drawVectorMountain = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple mountain
		 ctx.beginPath();
		 ctx.moveTo(x - 50, y + 30);
		 ctx.lineTo(x - 25, y - 30);
		 ctx.lineTo(x, y + 20);
		 ctx.lineTo(x + 25, y - 20);
		 ctx.lineTo(x + 50, y + 30);
		 ctx.fill();
	 };

	 const drawStar = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // 5-pointed star
		 ctx.beginPath();
		 for (let i = 0; i < 5; i++) {
			 const angle = (i * Math.PI * 2) / 5 - Math.PI / 2;
			 const outerX = x + Math.cos(angle) * 30;
			 const outerY = y + Math.sin(angle) * 30;
			 const innerX = x + Math.cos(angle + Math.PI / 5) * 12;
			 const innerY = y + Math.sin(angle + Math.PI / 5) * 12;
			 if (i === 0) ctx.moveTo(outerX, outerY);
			 else ctx.lineTo(outerX, outerY);
			 ctx.lineTo(innerX, innerY);
		 }
		 ctx.closePath();
		 ctx.fill();
	 };

	 const drawVectorStar = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple star
		 ctx.beginPath();
		 for (let i = 0; i < 5; i++) {
			 const angle = (i * Math.PI * 2) / 5 - Math.PI / 2;
			 const outerX = x + Math.cos(angle) * 35;
			 const outerY = y + Math.sin(angle) * 35;
			 const innerX = x + Math.cos(angle + Math.PI / 5) * 15;
			 const innerY = y + Math.sin(angle + Math.PI / 5) * 15;
			 if (i === 0) ctx.moveTo(outerX, outerY);
			 else ctx.lineTo(outerX, outerY);
			 ctx.lineTo(innerX, innerY);
		 }
		 ctx.closePath();
		 ctx.fill();
	 };

	 const drawHeart = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Heart shape
		 ctx.beginPath();
		 ctx.moveTo(x, y + 10);
		 ctx.bezierCurveTo(x - 20, y - 10, x - 30, y + 5, x, y + 25);
		 ctx.bezierCurveTo(x + 30, y + 5, x + 20, y - 10, x, y + 10);
		 ctx.fill();
	 };

	 const drawVectorHeart = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple heart
		 ctx.beginPath();
		 ctx.moveTo(x, y + 15);
		 ctx.bezierCurveTo(x - 25, y - 15, x - 35, y + 10, x, y + 30);
		 ctx.bezierCurveTo(x + 35, y + 10, x + 25, y - 15, x, y + 15);
		 ctx.fill();
	 };

	 const drawSun = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Sun center
		 ctx.beginPath();
		 ctx.arc(x, y, 20, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Sun rays
		 for (let i = 0; i < 8; i++) {
			 const angle = (i * Math.PI * 2) / 8;
			 const startX = x + Math.cos(angle) * 25;
			 const startY = y + Math.sin(angle) * 25;
			 const endX = x + Math.cos(angle) * 35;
			 const endY = y + Math.sin(angle) * 35;
			 ctx.beginPath();
			 ctx.moveTo(startX, startY);
			 ctx.lineTo(endX, endY);
			 ctx.lineWidth = 4;
			 ctx.stroke();
		 }
	 };

	 const drawVectorSun = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple sun
		 ctx.beginPath();
		 ctx.arc(x, y, 25, 0, 2 * Math.PI);
		 ctx.fill();
		 for (let i = 0; i < 8; i++) {
			 const angle = (i * Math.PI * 2) / 8;
			 const startX = x + Math.cos(angle) * 30;
			 const startY = y + Math.sin(angle) * 30;
			 const endX = x + Math.cos(angle) * 40;
			 const endY = y + Math.sin(angle) * 40;
			 ctx.beginPath();
			 ctx.moveTo(startX, startY);
			 ctx.lineTo(endX, endY);
			 ctx.lineWidth = 5;
			 ctx.stroke();
		 }
	 };

	 const drawMoon = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Moon crescent
		 ctx.beginPath();
		 ctx.arc(x, y, 25, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.fillStyle = '#000';
		 ctx.beginPath();
		 ctx.arc(x + 8, y, 20, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.fillStyle = color;
	 };

	 const drawVectorMoon = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple moon
		 ctx.beginPath();
		 ctx.arc(x, y, 30, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.fillStyle = '#000';
		 ctx.beginPath();
		 ctx.arc(x + 10, y, 25, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.fillStyle = color;
	 };

	 const drawCloud = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Cloud shape
		 ctx.beginPath();
		 ctx.arc(x - 15, y, 15, 0, 2 * Math.PI);
		 ctx.arc(x, y - 10, 20, 0, 2 * Math.PI);
		 ctx.arc(x + 15, y, 15, 0, 2 * Math.PI);
		 ctx.arc(x - 5, y + 5, 12, 0, 2 * Math.PI);
		 ctx.arc(x + 10, y + 5, 12, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorCloud = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple cloud
		 ctx.beginPath();
		 ctx.arc(x - 20, y, 18, 0, 2 * Math.PI);
		 ctx.arc(x, y - 15, 25, 0, 2 * Math.PI);
		 ctx.arc(x + 20, y, 18, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawBall = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Ball with highlight
		 ctx.beginPath();
		 ctx.arc(x, y, 25, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.fillStyle = '#fff';
		 ctx.beginPath();
		 ctx.arc(x - 8, y - 8, 8, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.fillStyle = color;
	 };

	 const drawVectorBall = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple ball
		 ctx.beginPath();
		 ctx.arc(x, y, 30, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawBook = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Book cover
		 ctx.fillRect(x - 20, y - 15, 40, 30);
		 
		 // Book pages
		 ctx.fillStyle = '#fff';
		 ctx.fillRect(x - 18, y - 13, 36, 26);
		 
		 // Book spine
		 ctx.fillStyle = color;
		 ctx.fillRect(x - 20, y - 15, 4, 30);
	 };

	 const drawVectorBook = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple book
		 ctx.fillRect(x - 25, y - 20, 50, 40);
		 ctx.fillStyle = '#fff';
		 ctx.fillRect(x - 23, y - 18, 46, 36);
		 ctx.fillStyle = color;
		 ctx.fillRect(x - 25, y - 20, 5, 40);
	 };

	 const drawCup = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Cup body
		 ctx.fillRect(x - 15, y - 10, 30, 20);
		 
		 // Cup handle
		 ctx.beginPath();
		 ctx.arc(x + 20, y, 8, 0, Math.PI);
		 ctx.lineWidth = 6;
		 ctx.stroke();
		 
		 // Steam
		 ctx.strokeStyle = '#666';
		 ctx.lineWidth = 2;
		 ctx.beginPath();
		 ctx.moveTo(x - 5, y - 15);
		 ctx.lineTo(x - 8, y - 25);
		 ctx.moveTo(x, y - 15);
		 ctx.lineTo(x + 3, y - 25);
		 ctx.moveTo(x + 5, y - 15);
		 ctx.lineTo(x + 8, y - 25);
		 ctx.stroke();
	 };

	 const drawVectorCup = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple cup
		 ctx.fillRect(x - 20, y - 15, 40, 30);
		 ctx.beginPath();
		 ctx.arc(x + 25, y, 10, 0, Math.PI);
		 ctx.lineWidth = 8;
		 ctx.stroke();
	 };

	 const drawBird = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Body
		 ctx.beginPath();
		 ctx.ellipse(x, y, 30, 20, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Head
		 ctx.beginPath();
		 ctx.arc(x - 20, y - 10, 15, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Beak
		 ctx.fillStyle = '#ffa500';
		 ctx.beginPath();
		 ctx.moveTo(x - 35, y - 10);
		 ctx.lineTo(x - 45, y - 5);
		 ctx.lineTo(x - 35, y);
		 ctx.fill();
		 
		 // Wings
		 ctx.fillStyle = color;
		 ctx.beginPath();
		 ctx.ellipse(x + 10, y - 5, 25, 15, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Tail
		 ctx.beginPath();
		 ctx.moveTo(x + 30, y);
		 ctx.lineTo(x + 50, y - 10);
		 ctx.lineTo(x + 50, y + 10);
		 ctx.fill();
	 };

	 const drawVectorBird = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple geometric bird
		 ctx.beginPath();
		 ctx.ellipse(x, y, 40, 25, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Head
		 ctx.beginPath();
		 ctx.arc(x - 25, y - 15, 18, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Beak
		 ctx.fillStyle = '#ffa500';
		 ctx.beginPath();
		 ctx.moveTo(x - 43, y - 15);
		 ctx.lineTo(x - 55, y - 10);
		 ctx.lineTo(x - 43, y - 5);
		 ctx.fill();
	 };

	 const drawAirplane = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Body
		 ctx.fillRect(x - 40, y - 5, 80, 10);
		 
		 // Wings
		 ctx.fillRect(x - 20, y - 20, 40, 8);
		 ctx.fillRect(x - 20, y + 12, 40, 8);
		 
		 // Tail
		 ctx.fillRect(x + 35, y - 15, 15, 8);
		 ctx.fillRect(x + 35, y + 7, 15, 8);
		 
		 // Nose
		 ctx.beginPath();
		 ctx.moveTo(x - 40, y);
		 ctx.lineTo(x - 50, y - 3);
		 ctx.lineTo(x - 50, y + 3);
		 ctx.fill();
	 };

	 const drawVectorAirplane = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Clean geometric airplane
		 ctx.fillRect(x - 50, y - 8, 100, 16);
		 ctx.fillRect(x - 25, y - 25, 50, 10);
		 ctx.fillRect(x - 25, y + 15, 50, 10);
		 ctx.fillRect(x + 40, y - 20, 20, 10);
		 ctx.fillRect(x + 40, y + 10, 20, 10);
	 };

	 const drawCar = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Body
		 ctx.fillRect(x - 40, y - 15, 80, 30);
		 
		 // Wheels
		 ctx.fillStyle = '#333';
		 ctx.beginPath();
		 ctx.arc(x - 25, y + 15, 8, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + 25, y + 15, 8, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Windows
		 ctx.fillStyle = '#87ceeb';
		 ctx.fillRect(x - 30, y - 10, 60, 15);
	 };

	 const drawVectorCar = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple car shape
		 ctx.fillRect(x - 50, y - 20, 100, 40);
		 ctx.fillStyle = '#333';
		 ctx.beginPath();
		 ctx.arc(x - 30, y + 20, 12, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + 30, y + 20, 12, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawTree = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Trunk
		 ctx.fillStyle = '#8b4513';
		 ctx.fillRect(x - 5, y + 20, 10, 40);
		 
		 // Leaves
		 ctx.fillStyle = color;
		 ctx.beginPath();
		 ctx.arc(x, y, 30, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorTree = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Geometric tree
		 ctx.fillStyle = '#8b4513';
		 ctx.fillRect(x - 8, y + 25, 16, 50);
		 ctx.fillStyle = color;
		 ctx.beginPath();
		 ctx.arc(x, y, 35, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawFlower = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Stem
		 ctx.fillStyle = '#228b22';
		 ctx.fillRect(x - 2, y + 20, 4, 40);
		 
		 // Petals
		 ctx.fillStyle = color;
		 for (let i = 0; i < 6; i++) {
			 const angle = (i * Math.PI * 2) / 6;
			 const px = x + Math.cos(angle) * 20;
			 const py = y + Math.sin(angle) * 20;
			 ctx.beginPath();
			 ctx.arc(px, py, 8, 0, 2 * Math.PI);
			 ctx.fill();
		 }
		 
		 // Center
		 ctx.fillStyle = '#ffd700';
		 ctx.beginPath();
		 ctx.arc(x, y, 8, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorFlower = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple flower
		 ctx.fillStyle = '#228b22';
		 ctx.fillRect(x - 3, y + 25, 6, 50);
		 ctx.fillStyle = color;
		 for (let i = 0; i < 5; i++) {
			 const angle = (i * Math.PI * 2) / 5;
			 const px = x + Math.cos(angle) * 25;
			 const py = y + Math.sin(angle) * 25;
			 ctx.beginPath();
			 ctx.arc(px, py, 10, 0, 2 * Math.PI);
			 ctx.fill();
		 }
	 };

	 const drawCat = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Head
		 ctx.beginPath();
		 ctx.arc(x, y, 25, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Ears
		 ctx.beginPath();
		 ctx.moveTo(x - 20, y - 20);
		 ctx.lineTo(x - 30, y - 35);
		 ctx.lineTo(x - 10, y - 25);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.moveTo(x + 20, y - 20);
		 ctx.lineTo(x + 30, y - 35);
		 ctx.lineTo(x + 10, y - 25);
		 ctx.fill();
		 
		 // Eyes
		 ctx.fillStyle = '#000';
		 ctx.beginPath();
		 ctx.arc(x - 10, y - 5, 3, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + 10, y - 5, 3, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Nose
		 ctx.fillStyle = '#ff69b4';
		 ctx.beginPath();
		 ctx.arc(x, y, 2, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorCat = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple cat
		 ctx.beginPath();
		 ctx.arc(x, y, 30, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.moveTo(x - 25, y - 25);
		 ctx.lineTo(x - 35, y - 40);
		 ctx.lineTo(x - 15, y - 30);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.moveTo(x + 25, y - 25);
		 ctx.lineTo(x + 35, y - 40);
		 ctx.lineTo(x + 15, y - 30);
		 ctx.fill();
	 };

	 const drawDog = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Head
		 ctx.beginPath();
		 ctx.arc(x, y, 25, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Ears
		 ctx.beginPath();
		 ctx.arc(x - 20, y - 15, 8, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + 20, y - 15, 8, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Eyes
		 ctx.fillStyle = '#000';
		 ctx.beginPath();
		 ctx.arc(x - 10, y - 5, 3, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + 10, y - 5, 3, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Nose
		 ctx.fillStyle = '#000';
		 ctx.beginPath();
		 ctx.arc(x, y, 2, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorDog = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple dog
		 ctx.beginPath();
		 ctx.arc(x, y, 30, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x - 25, y - 20, 10, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.arc(x + 25, y - 20, 10, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawFish = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Body
		 ctx.beginPath();
		 ctx.ellipse(x, y, 30, 20, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 
		 // Tail
		 ctx.beginPath();
		 ctx.moveTo(x + 30, y);
		 ctx.lineTo(x + 50, y - 15);
		 ctx.lineTo(x + 50, y + 15);
		 ctx.fill();
		 
		 // Fins
		 ctx.beginPath();
		 ctx.moveTo(x - 20, y - 15);
		 ctx.lineTo(x - 30, y - 25);
		 ctx.lineTo(x - 20, y - 20);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.moveTo(x - 20, y + 15);
		 ctx.lineTo(x - 30, y + 25);
		 ctx.lineTo(x - 20, y + 20);
		 ctx.fill();
		 
		 // Eye
		 ctx.fillStyle = '#000';
		 ctx.beginPath();
		 ctx.arc(x - 10, y - 5, 3, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorFish = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple fish
		 ctx.beginPath();
		 ctx.ellipse(x, y, 35, 25, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.moveTo(x + 35, y);
		 ctx.lineTo(x + 55, y - 20);
		 ctx.lineTo(x + 55, y + 20);
		 ctx.fill();
	 };

	 const drawButterfly = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Body
		 ctx.fillStyle = '#8b4513';
		 ctx.fillRect(x - 2, y - 30, 4, 60);
		 
		 // Wings
		 ctx.fillStyle = color;
		 ctx.beginPath();
		 ctx.ellipse(x - 20, y - 20, 25, 15, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x + 20, y - 20, 25, 15, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x - 15, y + 10, 20, 12, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x + 15, y + 10, 20, 12, 0, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawVectorButterfly = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple butterfly
		 ctx.fillStyle = '#8b4513';
		 ctx.fillRect(x - 3, y - 35, 6, 70);
		 ctx.fillStyle = color;
		 ctx.beginPath();
		 ctx.ellipse(x - 25, y - 25, 30, 20, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x + 25, y - 25, 30, 20, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x - 20, y + 15, 25, 15, 0, 0, 2 * Math.PI);
		 ctx.fill();
		 ctx.beginPath();
		 ctx.ellipse(x + 20, y + 15, 25, 15, 0, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawGenericShape = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Default shape
		 ctx.beginPath();
		 ctx.arc(x, y, 40, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const drawHouse = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // House body
		 ctx.fillRect(x - 30, y - 10, 60, 40);
		 
		 // Roof
		 ctx.beginPath();
		 ctx.moveTo(x - 35, y - 10);
		 ctx.lineTo(x, y - 25);
		 ctx.lineTo(x + 35, y - 10);
		 ctx.fill();
		 
		 // Door
		 ctx.fillStyle = '#8b4513';
		 ctx.fillRect(x - 8, y + 10, 16, 20);
		 
		 // Windows
		 ctx.fillStyle = '#87ceeb';
		 ctx.fillRect(x - 20, y - 5, 12, 12);
		 ctx.fillRect(x + 8, y - 5, 12, 12);
	 };

	 const drawVectorHouse = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Simple house
		 ctx.fillRect(x - 40, y - 15, 80, 50);
		 ctx.beginPath();
		 ctx.moveTo(x - 45, y - 15);
		 ctx.lineTo(x, y - 30);
		 ctx.lineTo(x + 45, y - 15);
		 ctx.fill();
	 };

	 const drawVectorGenericShape = (ctx: CanvasRenderingContext2D, x: number, y: number, color: string) => {
		 // Default vector shape
		 ctx.beginPath();
		 ctx.arc(x, y, 45, 0, 2 * Math.PI);
		 ctx.fill();
	 };

	 const handleGenVector = async () => {
		 if (!prompt.trim()) {
			 console.log('No prompt provided');
			 return;
		 }
		 
		 console.log('🚀 Starting vector generation for prompt:', prompt);
		 try {
			 const generatedImage = await generateImageFromPrompt(prompt, 'vector');
			 console.log('✅ Generated image successfully:', generatedImage.substring(0, 50) + '...');
			 setPreview(generatedImage);
			 setHistoryImgs(h => [generatedImage, ...h].slice(0, 20));
		 } catch (error) {
			 console.error('❌ Error generating vector:', error);
			 alert('Error generating image: ' + error.message);
		 }
	 };

	 const handleGenPng = async () => {
		 if (!prompt.trim()) {
			 console.log('No prompt provided');
			 return;
		 }
		 
		 console.log('🚀 Starting PNG generation for prompt:', prompt);
		 try {
			 const generatedImage = await generateImageFromPrompt(prompt, 'png');
			 console.log('✅ Generated image successfully:', generatedImage.substring(0, 50) + '...');
			 setPreview(generatedImage);
			 setHistoryImgs(h => [generatedImage, ...h].slice(0, 20));
		 } catch (error) {
			 console.error('❌ Error generating PNG:', error);
			 alert('Error generating image: ' + error.message);
		 }
	 };

	 return (
		 <main className="min-h-screen w-full bg-black flex items-start justify-center overflow-auto">
			 <div style={{ width: baseW, height: baseH, transform: `scale(${scale})`, transformOrigin: 'top center' }} className="relative overflow-hidden" onMouseMove={onMove} onMouseUp={endDrag}>
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

				 {/* Prompt */}
				 <div className={`absolute z-20 overflow-auto scrollbox ${calib ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.prompt), borderRadius: 16 }} onMouseDown={(e)=>startDrag(e,'prompt')}>
					 <textarea className="w-full h-full bg-transparent outline-none resize-none border-0 scrollbox" style={{ fontSize: '13px', color: '#000080', padding: '6px 10px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere', boxSizing: 'border-box' }} value={prompt} onChange={(e)=>setPrompt(e.target.value)} wrap="soft" />
				 </div>

				 {/* Buttons */}
				 <button className={`absolute z-20 ${calib ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.vecBtn), background: 'transparent', cursor: 'pointer' }} onMouseDown={(e)=>startDrag(e,'vecBtn')} onClick={handleGenVector} aria-label="Gerar Vetor" />
				 <button className={`absolute z-20 ${calib ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.pngBtn), background: 'transparent', cursor: 'pointer' }} onMouseDown={(e)=>startDrag(e,'pngBtn')} onClick={handleGenPng} aria-label="Gerar PNG" />

				 {/* History */}
				 <div className={`absolute z-20 overflow-auto scrollbox ${calib ? 'outline outline-2 outline-yellow-400' : ''}`} style={{ ...toStyle(rects.hist) }} onMouseDown={(e)=>startDrag(e,'hist')}>
					 <div className="grid grid-cols-1 gap-2">
						 {historyImgs.map((src, idx) => (
							 <img key={idx} src={src} alt={"hist-"+idx} className="w-full h-24 object-cover rounded cursor-pointer" onClick={() => setPreview(src)} />
						 ))}
					 </div>
				 </div>


				 {/* Calibrate toggle */}
				 <button className="absolute z-30 bottom-2 right-2 bg-white/70 text-black px-3 py-1 rounded" onClick={()=>setCalib(v=>!v)}>{calib ? 'Lock' : 'Calibrate'}</button>
				 
				 {/* Advanced Editor Link */}
				 <button 
					 className="absolute z-30 bottom-2 left-2 bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition-colors" 
					 onClick={() => window.location.href = '/advanced'}
				 >
					 Advanced Studio
				 </button>
				  {/* Stage 2 Link */}
				  <button 
					  className="absolute z-30 bottom-2 left-40 bg-indigo-600 text-white px-3 py-1 rounded hover:bg-indigo-700 transition-colors" 
					  onClick={() => window.location.href = '/stage2'}
				  >
					  Stage 2
				  </button>
			 </div>
		 </main>
	 );
}
