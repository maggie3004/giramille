"use client";

import React, { useState, useRef, useEffect } from 'react';

export interface SceneNode {
  id: string;
  type: 'image' | 'vector' | 'text' | 'shape';
  name: string;
  visible: boolean;
  locked: boolean;
  opacity: number;
  transform: {
    x: number;
    y: number;
    scaleX: number;
    scaleY: number;
    rotation: number;
  };
  mask?: {
    type: 'rect' | 'circle' | 'path';
    data: any;
  };
  content: {
    src?: string; // for images
    svg?: string; // for vectors
    text?: string; // for text
    style?: any; // for text styling
  };
  children?: SceneNode[];
}

export interface SceneGraphProps {
  nodes: SceneNode[];
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  onNodeUpdate: (nodeId: string, updates: Partial<SceneNode>) => void;
  onNodeAdd: (parentId: string | null, node: Omit<SceneNode, 'id'>) => void;
  onNodeRemove: (nodeId: string) => void;
  onNodeReorder: (nodeId: string, newIndex: number) => void;
}

export default function SceneGraph({
  nodes,
  selectedNodeId,
  onNodeSelect,
  onNodeUpdate,
  onNodeAdd,
  onNodeRemove,
  onNodeReorder
}: SceneGraphProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const dragRef = useRef<HTMLDivElement | null>(null);

  const toggleExpanded = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const handleDragStart = (e: React.DragEvent, nodeId: string) => {
    setDraggedNode(nodeId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, targetNodeId: string) => {
    e.preventDefault();
    if (draggedNode && draggedNode !== targetNodeId) {
      // Find the target node and reorder
      const targetIndex = nodes.findIndex(n => n.id === targetNodeId);
      if (targetIndex !== -1) {
        onNodeReorder(draggedNode, targetIndex);
      }
    }
    setDraggedNode(null);
  };

  const renderNode = (node: SceneNode, depth: number = 0) => {
    const isExpanded = expandedNodes.has(node.id);
    const isSelected = selectedNodeId === node.id;
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.id} className="select-none">
        <div
          className={`flex items-center py-1 px-2 cursor-pointer hover:bg-gray-100 rounded ${
            isSelected ? 'bg-blue-100 border-l-2 border-blue-500' : ''
          }`}
          style={{ paddingLeft: `${depth * 20 + 8}px` }}
          onClick={() => onNodeSelect(node.id)}
          draggable
          onDragStart={(e) => handleDragStart(e, node.id)}
          onDragOver={handleDragOver}
          onDrop={(e) => handleDrop(e, node.id)}
        >
          {/* Expand/Collapse Button */}
          <button
            className="w-4 h-4 mr-2 flex items-center justify-center"
            onClick={(e) => {
              e.stopPropagation();
              if (hasChildren) toggleExpanded(node.id);
            }}
          >
            {hasChildren ? (isExpanded ? '▼' : '▶') : '•'}
          </button>

          {/* Visibility Toggle */}
          <button
            className="w-4 h-4 mr-2 flex items-center justify-center"
            onClick={(e) => {
              e.stopPropagation();
              onNodeUpdate(node.id, { visible: !node.visible });
            }}
          >
            {node.visible ? '👁' : '👁‍🗨'}
          </button>

          {/* Lock Toggle */}
          <button
            className="w-4 h-4 mr-2 flex items-center justify-center"
            onClick={(e) => {
              e.stopPropagation();
              onNodeUpdate(node.id, { locked: !node.locked });
            }}
          >
            {node.locked ? '🔒' : '🔓'}
          </button>

          {/* Node Type Icon */}
          <span className="mr-2">
            {node.type === 'image' ? '🖼️' : 
             node.type === 'vector' ? '📐' : 
             node.type === 'text' ? '📝' : '🔷'}
          </span>

          {/* Node Name */}
          <span className="flex-1 text-sm truncate">{node.name}</span>

          {/* Opacity Slider */}
          <input
            type="range"
            min="0"
            max="100"
            value={node.opacity}
            className="w-16 h-2 mr-2"
            onChange={(e) => {
              e.stopPropagation();
              onNodeUpdate(node.id, { opacity: parseInt(e.target.value) });
            }}
          />

          {/* Delete Button */}
          <button
            className="w-4 h-4 text-red-500 hover:text-red-700"
            onClick={(e) => {
              e.stopPropagation();
              onNodeRemove(node.id);
            }}
          >
            🗑️
          </button>
        </div>

        {/* Children */}
        {hasChildren && isExpanded && (
          <div>
            {node.children!.map(child => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full h-full bg-white border border-gray-300 rounded-lg overflow-auto">
      <div className="p-2 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700">Scene Layers</h3>
      </div>
      <div className="p-1">
        {nodes.length === 0 ? (
          <div className="text-center text-gray-500 py-4 text-sm">
            No layers yet. Generate an image to start editing.
          </div>
        ) : (
          nodes.map(node => renderNode(node))
        )}
      </div>
    </div>
  );
}
