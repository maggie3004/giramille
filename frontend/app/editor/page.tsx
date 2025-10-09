"use client";

import React, { useState, useRef, useCallback } from 'react';
import SceneGraph, { SceneNode } from '../../components/SceneGraph';
import CanvasEditor from '../../components/CanvasEditor';
import AssetUploader from '../../components/AssetUploader';
import MultiViewGenerator from '../../components/MultiViewGenerator';

export default function AdvancedEditor() {
  const [nodes, setNodes] = useState<SceneNode[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'layers' | 'assets' | 'multiview'>('layers');
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const canvasRef = useRef<HTMLDivElement>(null);

  // Scene Graph Operations
  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const handleNodeUpdate = useCallback((nodeId: string, updates: Partial<SceneNode>) => {
    setNodes(prev => prev.map(node => 
      node.id === nodeId ? { ...node, ...updates } : node
    ));
  }, []);

  const handleNodeAdd = useCallback((parentId: string | null, newNode: Omit<SceneNode, 'id'>) => {
    const node: SceneNode = {
      ...newNode,
      id: Math.random().toString(36).substr(2, 9)
    };
    
    if (parentId) {
      setNodes(prev => prev.map(n => 
        n.id === parentId 
          ? { ...n, children: [...(n.children || []), node] }
          : n
      ));
    } else {
      setNodes(prev => [...prev, node]);
    }
  }, []);

  const handleNodeRemove = useCallback((nodeId: string) => {
    setNodes(prev => prev.filter(node => node.id !== nodeId));
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
    }
  }, [selectedNodeId]);

  const handleNodeReorder = useCallback((nodeId: string, newIndex: number) => {
    setNodes(prev => {
      const nodeIndex = prev.findIndex(n => n.id === nodeId);
      if (nodeIndex === -1) return prev;
      
      const node = prev[nodeIndex];
      const newNodes = prev.filter(n => n.id !== nodeId);
      newNodes.splice(newIndex, 0, node);
      return newNodes;
    });
  }, []);

  // Asset Operations
  const handleAssetUpload = useCallback(async (file: File, type: 'image' | 'vector') => {
    try {
      console.log('🔄 Uploading asset to backend...');
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', type);
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.asset) {
        const newNode: Omit<SceneNode, 'id'> = {
          type: type === 'image' ? 'image' : 'vector',
          name: file.name,
          visible: true,
          locked: false,
          opacity: 100,
          transform: {
            x: Math.random() * (canvasSize.width - 100),
            y: Math.random() * (canvasSize.height - 100),
            scaleX: 1,
            scaleY: 1,
            rotation: 0
          },
          content: {
            src: data.asset.url,
            svg: type === 'vector' ? data.asset.url : undefined
          }
        };
        handleNodeAdd(null, newNode);
        console.log('✅ Asset uploaded successfully');
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    } catch (error) {
      console.error('❌ Asset upload error:', error);
      alert('Error uploading asset: ' + (error as Error).message);
    }
  }, [canvasSize, handleNodeAdd]);

  const handleAssetInsert = useCallback((assetId: string, position: { x: number; y: number }) => {
    // This would be called when dragging an asset to the canvas
    console.log('Insert asset:', assetId, 'at position:', position);
  }, []);

  // Multi-view Operations
  const handleViewGenerated = useCallback(async (sourceImage: string, angle: string) => {
    try {
      console.log('🔄 Generating multi-view from backend...');
      
      const response = await fetch('/api/multiview/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: sourceImage,
          angle: angle,
          quality: 'high'
        })
      });

      if (!response.ok) {
        throw new Error(`Multi-view generation failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.view) {
        const newNode: Omit<SceneNode, 'id'> = {
          type: 'image',
          name: `Generated ${angle} view`,
          visible: true,
          locked: false,
          opacity: 100,
          transform: {
            x: Math.random() * (canvasSize.width - 100),
            y: Math.random() * (canvasSize.height - 100),
            scaleX: 1,
            scaleY: 1,
            rotation: 0
          },
          content: {
            src: data.view
          }
        };
        handleNodeAdd(null, newNode);
        console.log('✅ Multi-view generated successfully');
      } else {
        throw new Error(data.error || 'Multi-view generation failed');
      }
    } catch (error) {
      console.error('❌ Multi-view generation error:', error);
      alert('Error generating multi-view: ' + (error as Error).message);
    }
  }, [canvasSize, handleNodeAdd]);

  // Canvas Operations
  const handleCanvasDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    try {
      const assetData = JSON.parse(e.dataTransfer.getData('application/json'));
      const newNode: Omit<SceneNode, 'id'> = {
        type: assetData.type,
        name: assetData.name,
        visible: true,
        locked: false,
        opacity: 100,
        transform: {
          x: x - 50,
          y: y - 50,
          scaleX: 1,
          scaleY: 1,
          rotation: 0
        },
        content: {
          src: assetData.src
        }
      };
      handleNodeAdd(null, newNode);
    } catch (error) {
      console.error('Error parsing dropped asset:', error);
    }
  }, [handleNodeAdd]);

  const handleCanvasDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  // Scene Export Operations
  const handleExportScene = useCallback(async () => {
    try {
      console.log('🔄 Exporting scene to backend...');
      
      const sceneData = {
        nodes: nodes,
        canvasSize: canvasSize,
        timestamp: new Date().toISOString()
      };
      
      const response = await fetch('/api/export/scene', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sceneData)
      });

      if (!response.ok) {
        throw new Error(`Scene export failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.export) {
        // Download the exported file
        const link = document.createElement('a');
        link.href = data.export.url;
        link.download = data.export.filename || 'scene-export.zip';
        link.click();
        console.log('✅ Scene exported successfully');
      } else {
        throw new Error(data.error || 'Scene export failed');
      }
    } catch (error) {
      console.error('❌ Scene export error:', error);
      alert('Error exporting scene: ' + (error as Error).message);
    }
  }, [nodes, canvasSize]);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-800">Advanced Image Editor</h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Canvas Size:</label>
              <input
                type="number"
                value={canvasSize.width}
                onChange={(e) => setCanvasSize(prev => ({ ...prev, width: parseInt(e.target.value) || 800 }))}
                className="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
              />
              <span className="text-gray-500">×</span>
              <input
                type="number"
                value={canvasSize.height}
                onChange={(e) => setCanvasSize(prev => ({ ...prev, height: parseInt(e.target.value) || 600 }))}
                className="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            <button
              onClick={handleExportScene}
              disabled={nodes.length === 0}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              Export Scene
            </button>
            <button
              onClick={() => window.location.href = '/'}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              Back to Main
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          {/* Tab Navigation */}
          <div className="flex border-b border-gray-200">
            <button
              className={`flex-1 px-4 py-3 text-sm font-medium ${
                activeTab === 'layers' 
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              onClick={() => setActiveTab('layers')}
            >
              Layers
            </button>
            <button
              className={`flex-1 px-4 py-3 text-sm font-medium ${
                activeTab === 'assets' 
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              onClick={() => setActiveTab('assets')}
            >
              Assets
            </button>
            <button
              className={`flex-1 px-4 py-3 text-sm font-medium ${
                activeTab === 'multiview' 
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              onClick={() => setActiveTab('multiview')}
            >
              Multi-View
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'layers' && (
              <SceneGraph
                nodes={nodes}
                selectedNodeId={selectedNodeId}
                onNodeSelect={handleNodeSelect}
                onNodeUpdate={handleNodeUpdate}
                onNodeAdd={handleNodeAdd}
                onNodeRemove={handleNodeRemove}
                onNodeReorder={handleNodeReorder}
              />
            )}
            {activeTab === 'assets' && (
              <div className="p-4">
                <AssetUploader
                  onAssetUpload={handleAssetUpload}
                  onAssetInsert={handleAssetInsert}
                />
              </div>
            )}
            {activeTab === 'multiview' && (
              <div className="p-4">
                <MultiViewGenerator
                  sourceImage={selectedNodeId ? nodes.find(n => n.id === selectedNodeId)?.content.src || null : null}
                  onViewGenerated={handleViewGenerated}
                />
              </div>
            )}
          </div>
        </div>

        {/* Main Canvas Area */}
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="bg-white rounded-lg shadow-lg p-4">
            <div
              ref={canvasRef}
              onDrop={handleCanvasDrop}
              onDragOver={handleCanvasDragOver}
              className="relative"
            >
              <CanvasEditor
                nodes={nodes}
                selectedNodeId={selectedNodeId}
                onNodeUpdate={handleNodeUpdate}
                onNodeSelect={handleNodeSelect}
                canvasWidth={canvasSize.width}
                canvasHeight={canvasSize.height}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
