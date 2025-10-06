"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { SceneNode } from './SceneGraph';

export interface CanvasEditorProps {
  nodes: SceneNode[];
  selectedNodeId: string | null;
  onNodeUpdate: (nodeId: string, updates: Partial<SceneNode>) => void;
  onNodeSelect: (nodeId: string | null) => void;
  canvasWidth: number;
  canvasHeight: number;
}

export default function CanvasEditor({
  nodes,
  selectedNodeId,
  onNodeUpdate,
  onNodeSelect,
  canvasWidth,
  canvasHeight
}: CanvasEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isResizing, setIsResizing] = useState(false);
  const [resizeHandle, setResizeHandle] = useState<string | null>(null);

  // Render all nodes to canvas
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Render nodes in order
    nodes.forEach(node => {
      if (!node.visible) return;

      ctx.save();
      
      // Apply opacity
      ctx.globalAlpha = node.opacity / 100;
      
      // Apply transform
      ctx.translate(node.transform.x, node.transform.y);
      ctx.rotate((node.transform.rotation * Math.PI) / 180);
      ctx.scale(node.transform.scaleX, node.transform.scaleY);

      // Render based on type
      if (node.type === 'image' && node.content.src) {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0);
        };
        img.src = node.content.src;
      } else if (node.type === 'vector' && node.content.svg) {
        // For SVG, we'll need to convert to canvas or use a different approach
        // This is a placeholder - in production, you'd use a library like fabric.js
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, 100, 100);
      } else if (node.type === 'text' && node.content.text) {
        ctx.fillStyle = node.content.style?.color || '#000';
        ctx.font = `${node.content.style?.fontSize || 16}px ${node.content.style?.fontFamily || 'Arial'}`;
        ctx.fillText(node.content.text, 0, 0);
      }

      ctx.restore();
    });
  }, [nodes, canvasWidth, canvasHeight]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on a node
    const clickedNode = findNodeAtPosition(x, y);
    if (clickedNode) {
      onNodeSelect(clickedNode.id);
      
      // Check if clicking on resize handle
      const handle = getResizeHandle(clickedNode, x, y);
      if (handle) {
        setIsResizing(true);
        setResizeHandle(handle);
        setDragStart({ x, y });
        return;
      }

      // Start dragging
      setIsDragging(true);
      setDragStart({ x, y });
    } else {
      // Deselect
      onNodeSelect(null);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!selectedNodeId) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDragging) {
      const dx = x - dragStart.x;
      const dy = y - dragStart.y;
      
      const selectedNode = nodes.find(n => n.id === selectedNodeId);
      if (selectedNode) {
        onNodeUpdate(selectedNodeId, {
          transform: {
            ...selectedNode.transform,
            x: selectedNode.transform.x + dx,
            y: selectedNode.transform.y + dy
          }
        });
      }
      setDragStart({ x, y });
    } else if (isResizing && resizeHandle) {
      const dx = x - dragStart.x;
      const dy = y - dragStart.y;
      
      const selectedNode = nodes.find(n => n.id === selectedNodeId);
      if (selectedNode) {
        let newScaleX = selectedNode.transform.scaleX;
        let newScaleY = selectedNode.transform.scaleY;

        if (resizeHandle.includes('right')) newScaleX += dx / 100;
        if (resizeHandle.includes('left')) newScaleX -= dx / 100;
        if (resizeHandle.includes('bottom')) newScaleY += dy / 100;
        if (resizeHandle.includes('top')) newScaleY -= dy / 100;

        onNodeUpdate(selectedNodeId, {
          transform: {
            ...selectedNode.transform,
            scaleX: Math.max(0.1, newScaleX),
            scaleY: Math.max(0.1, newScaleY)
          }
        });
      }
      setDragStart({ x, y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
    setResizeHandle(null);
  };

  const findNodeAtPosition = (x: number, y: number): SceneNode | null => {
    // Simple collision detection - in production, you'd use more sophisticated methods
    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      if (!node.visible) continue;

      const nodeX = node.transform.x;
      const nodeY = node.transform.y;
      const nodeWidth = 100 * node.transform.scaleX; // Placeholder width
      const nodeHeight = 100 * node.transform.scaleY; // Placeholder height

      if (x >= nodeX && x <= nodeX + nodeWidth && y >= nodeY && y <= nodeY + nodeHeight) {
        return node;
      }
    }
    return null;
  };

  const getResizeHandle = (node: SceneNode, x: number, y: number): string | null => {
    const nodeX = node.transform.x;
    const nodeY = node.transform.y;
    const nodeWidth = 100 * node.transform.scaleX;
    const nodeHeight = 100 * node.transform.scaleY;
    const handleSize = 8;

    // Check corners and edges
    if (x >= nodeX + nodeWidth - handleSize && y >= nodeY + nodeHeight - handleSize) {
      return 'bottom-right';
    }
    if (x >= nodeX && x <= nodeX + handleSize && y >= nodeY && y <= nodeY + handleSize) {
      return 'top-left';
    }
    // Add more handles as needed
    return null;
  };

  const renderSelectionBox = () => {
    if (!selectedNodeId) return null;

    const selectedNode = nodes.find(n => n.id === selectedNodeId);
    if (!selectedNode) return null;

    return (
      <div
        className="absolute border-2 border-blue-500 pointer-events-none"
        style={{
          left: selectedNode.transform.x,
          top: selectedNode.transform.y,
          width: 100 * selectedNode.transform.scaleX,
          height: 100 * selectedNode.transform.scaleY,
          transform: `rotate(${selectedNode.transform.rotation}deg)`
        }}
      >
        {/* Resize handles */}
        <div className="absolute -top-1 -left-1 w-2 h-2 bg-blue-500 border border-white cursor-nw-resize" />
        <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 border border-white cursor-ne-resize" />
        <div className="absolute -bottom-1 -left-1 w-2 h-2 bg-blue-500 border border-white cursor-sw-resize" />
        <div className="absolute -bottom-1 -right-1 w-2 h-2 bg-blue-500 border border-white cursor-se-resize" />
      </div>
    );
  };

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        className="border border-gray-300 cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      {renderSelectionBox()}
    </div>
  );
}
