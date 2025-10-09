'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';

// Types for the advanced scene editor
interface Vector2 {
  x: number;
  y: number;
}

interface Transform {
  position: Vector2;
  rotation: number;
  scale: Vector2;
}

interface SceneObject {
  id: string;
  type: string;
  transform: Transform;
  style: ObjectStyle;
  layer: number;
  visible: boolean;
  locked: boolean;
}

interface ObjectStyle {
  fillColor: string;
  strokeColor: string;
  strokeWidth: number;
  opacity: number;
  shadow?: ShadowStyle;
}

interface ShadowStyle {
  offset: Vector2;
  blur: number;
  color: string;
}

interface Scene {
  objects: SceneObject[];
  background: BackgroundStyle;
  canvasSize: Vector2;
}

interface BackgroundStyle {
  type: 'color' | 'gradient' | 'image';
  value: string;
}

interface ViewAngle {
  name: string;
  rotation: number;
  elevation: number;
}

// Advanced Scene Editor Component
export default function AdvancedSceneEditor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scene, setScene] = useState<Scene>({
    objects: [],
    background: { type: 'color', value: '#ffffff' },
    canvasSize: { x: 1024, y: 1024 }
  });
  
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<Vector2>({ x: 0, y: 0 });
  const [tool, setTool] = useState<'select' | 'add' | 'transform'>('select');
  const [showLayers, setShowLayers] = useState(true);
  const [showAssets, setShowAssets] = useState(false);

  // Object generation functions
  const generateObject = useCallback((type: string, position: Vector2): SceneObject => {
    const id = `obj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const baseStyle: ObjectStyle = {
      fillColor: '#4ecdc4',
      strokeColor: '#2c2c54',
      strokeWidth: 2,
      opacity: 1
    };

    // Style based on object type
    switch (type) {
      case 'house':
        return {
          id,
          type,
          transform: { position, rotation: 0, scale: { x: 1, y: 1 } },
          style: { ...baseStyle, fillColor: '#ff6b6b' },
          layer: scene.objects.length,
          visible: true,
          locked: false
        };
      case 'car':
        return {
          id,
          type,
          transform: { position, rotation: 0, scale: { x: 1, y: 1 } },
          style: { ...baseStyle, fillColor: '#3742fa' },
          layer: scene.objects.length,
          visible: true,
          locked: false
        };
      case 'tree':
        return {
          id,
          type,
          transform: { position, rotation: 0, scale: { x: 1, y: 1 } },
          style: { ...baseStyle, fillColor: '#2ed573' },
          layer: scene.objects.length,
          visible: true,
          locked: false
        };
      default:
        return {
          id,
          type,
          transform: { position, rotation: 0, scale: { x: 1, y: 1 } },
          style: baseStyle,
          layer: scene.objects.length,
          visible: true,
          locked: false
        };
    }
  }, [scene.objects.length]);

  // Add object to scene
  const addObjectToScene = useCallback((objectType: string, position: Vector2) => {
    const newObject = generateObject(objectType, position);
    setScene(prev => ({
      ...prev,
      objects: [...prev.objects, newObject]
    }));
  }, [generateObject]);

  // Remove object from scene
  const removeObjectFromScene = useCallback((objectId: string) => {
    setScene(prev => ({
      ...prev,
      objects: prev.objects.filter(obj => obj.id !== objectId)
    }));
    if (selectedObject === objectId) {
      setSelectedObject(null);
    }
  }, [selectedObject]);

  // Transform object
  const transformObject = useCallback((objectId: string, transform: Partial<Transform>) => {
    setScene(prev => ({
      ...prev,
      objects: prev.objects.map(obj => 
        obj.id === objectId 
          ? { ...obj, transform: { ...obj.transform, ...transform } }
          : obj
      )
    }));
  }, []);

  // Update object style
  const updateObjectStyle = useCallback((objectId: string, style: Partial<ObjectStyle>) => {
    setScene(prev => ({
      ...prev,
      objects: prev.objects.map(obj => 
        obj.id === objectId 
          ? { ...obj, style: { ...obj.style, ...style } }
          : obj
      )
    }));
  }, []);

  // Canvas drawing functions
  const drawScene = useCallback((ctx: CanvasRenderingContext2D) => {
    // Clear canvas
    ctx.clearRect(0, 0, scene.canvasSize.x, scene.canvasSize.y);
    
    // Draw background
    drawBackground(ctx);
    
    // Draw objects (sorted by layer)
    const sortedObjects = [...scene.objects].sort((a, b) => a.layer - b.layer);
    sortedObjects.forEach(obj => {
      if (obj.visible) {
        drawObject(ctx, obj);
      }
    });
    
    // Draw selection outline
    if (selectedObject) {
      const obj = scene.objects.find(o => o.id === selectedObject);
      if (obj) {
        drawSelectionOutline(ctx, obj);
      }
    }
  }, [scene, selectedObject]);

  const drawBackground = (ctx: CanvasRenderingContext2D) => {
    const { background } = scene;
    
    switch (background.type) {
      case 'color':
        ctx.fillStyle = background.value;
        ctx.fillRect(0, 0, scene.canvasSize.x, scene.canvasSize.y);
        break;
      case 'gradient':
        const gradient = ctx.createLinearGradient(0, 0, scene.canvasSize.x, scene.canvasSize.y);
        gradient.addColorStop(0, background.value.split(',')[0]);
        gradient.addColorStop(1, background.value.split(',')[1] || background.value);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, scene.canvasSize.x, scene.canvasSize.y);
        break;
    }
  };

  const drawObject = (ctx: CanvasRenderingContext2D, obj: SceneObject) => {
    ctx.save();
    
    // Apply transform
    ctx.translate(obj.transform.position.x, obj.transform.position.y);
    ctx.rotate(obj.transform.rotation);
    ctx.scale(obj.transform.scale.x, obj.transform.scale.y);
    
    // Apply style
    ctx.fillStyle = obj.style.fillColor;
    ctx.strokeStyle = obj.style.strokeColor;
    ctx.lineWidth = obj.style.strokeWidth;
    ctx.globalAlpha = obj.style.opacity;
    
    // Draw object based on type
    switch (obj.type) {
      case 'house':
        drawHouse(ctx);
        break;
      case 'car':
        drawCar(ctx);
        break;
      case 'tree':
        drawTree(ctx);
        break;
      default:
        // If the object type is not recognized, do nothing for now.
        break;
    }
    
    ctx.restore();
  };

  const drawHouse = (ctx: CanvasRenderingContext2D) => {
    // House base
    ctx.fillRect(-50, -30, 100, 60);
    
    // Roof
    ctx.beginPath();
    ctx.moveTo(-50, -30);
    ctx.lineTo(0, -60);
    ctx.lineTo(50, -30);
    ctx.closePath();
    ctx.fill();
    
    // Door
    ctx.fillStyle = '#8b4513';
    ctx.fillRect(-10, 0, 20, 30);
    
    // Windows
    ctx.fillStyle = '#87ceeb';
    ctx.fillRect(-30, -20, 15, 15);
    ctx.fillRect(15, -20, 15, 15);
  };

  const drawCar = (ctx: CanvasRenderingContext2D) => {
    // Car body
    ctx.fillRect(-60, -20, 120, 40);
    
    // Wheels
    ctx.fillStyle = '#2c2c54';
    ctx.beginPath();
    ctx.arc(-40, 20, 15, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(40, 20, 15, 0, 2 * Math.PI);
    ctx.fill();
    
    // Windows
    ctx.fillStyle = '#87ceeb';
    ctx.fillRect(-40, -15, 30, 15);
    ctx.fillRect(10, -15, 30, 15);
  };

  const drawTree = (ctx: CanvasRenderingContext2D) => {
    // Trunk
    ctx.fillStyle = '#8b4513';
    ctx.fillRect(-5, -20, 10, 40);
    
    // Leaves
    ctx.fillStyle = '#228B22'; // Default green color for leaves
    ctx.beginPath();
    ctx.arc(0, -30, 25, 0, 2 * Math.PI);
    ctx.fill();
  };

  const drawGenericShape = (ctx: CanvasRenderingContext2D) => {
    ctx.beginPath();
    ctx.arc(0, 0, 30, 0, 2 * Math.PI);
    ctx.fill();
  };

  const drawSelectionOutline = (ctx: CanvasRenderingContext2D, obj: SceneObject) => {
    ctx.save();
    ctx.translate(obj.transform.position.x, obj.transform.position.y);
    ctx.rotate(obj.transform.rotation);
    ctx.scale(obj.transform.scale.x, obj.transform.scale.y);
    
    ctx.strokeStyle = '#007bff';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(-60, -60, 120, 120);
    ctx.setLineDash([]);
    
    ctx.restore();
  };

  // Mouse event handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (tool === 'select') {
      // Find object at click position
      const clickedObject = findObjectAtPosition({ x, y });
      setSelectedObject(clickedObject?.id || null);
      
      if (clickedObject) {
        setIsDragging(true);
        setDragStart({ x, y });
      }
    } else if (tool === 'add') {
      // Add new object at click position
      const objectType = prompt('Enter object type (house, car, tree):') || 'house';
      addObjectToScene(objectType, { x, y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging && selectedObject) {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const deltaX = x - dragStart.x;
      const deltaY = y - dragStart.y;
      
      transformObject(selectedObject, {
        position: {
          x: scene.objects.find(obj => obj.id === selectedObject)!.transform.position.x + deltaX,
          y: scene.objects.find(obj => obj.id === selectedObject)!.transform.position.y + deltaY
        }
      });
      
      setDragStart({ x, y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const findObjectAtPosition = (position: Vector2): SceneObject | null => {
    // Simple bounding box collision detection
    for (let i = scene.objects.length - 1; i >= 0; i--) {
      const obj = scene.objects[i];
      const bounds = {
        left: obj.transform.position.x - 50,
        right: obj.transform.position.x + 50,
        top: obj.transform.position.y - 50,
        bottom: obj.transform.position.y + 50
      };
      
      if (position.x >= bounds.left && position.x <= bounds.right &&
          position.y >= bounds.top && position.y <= bounds.bottom) {
        return obj;
      }
    }
    return null;
  };

  // Multi-view generation
  const generateMultiView = useCallback(async (angles: ViewAngle[]) => {
    const results = [];
    
    for (const angle of angles) {
      // Simulate AI generation for different angles
      const result = await generateFromAngle(scene, angle);
      results.push({ angle, result });
    }
    
    return results;
  }, [scene]);

  const generateFromAngle = async (scene: Scene, angle: ViewAngle): Promise<string> => {
    // This would integrate with the AI generation system
    // For now, return a placeholder
    return `Generated view from angle: ${angle.name}`;
  };

  // Asset upload handler
  const handleAssetUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Process and integrate asset
      processAsset(file);
    }
  };

  const processAsset = async (file: File) => {
    // This would integrate with the asset processing pipeline
    console.log('Processing asset:', file.name);
  };

  // Render scene on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    drawScene(ctx);
  }, [drawScene]);

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Tool Palette */}
      <div className="w-64 bg-white border-r p-4">
        <h3 className="text-lg font-semibold mb-4">Tools</h3>
        
        <div className="space-y-2 mb-6">
          <button
            className={`w-full p-2 rounded ${tool === 'select' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setTool('select')}
          >
            Select
          </button>
          <button
            className={`w-full p-2 rounded ${tool === 'add' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setTool('add')}
          >
            Add Object
          </button>
          <button
            className={`w-full p-2 rounded ${tool === 'transform' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setTool('transform')}
          >
            Transform
          </button>
        </div>

        {/* Object Library */}
        <div className="mb-6">
          <h4 className="font-medium mb-2">Objects</h4>
          <div className="grid grid-cols-2 gap-2">
            {['house', 'car', 'tree', 'person', 'animal'].map(type => (
              <button
                key={type}
                className="p-2 bg-gray-100 rounded hover:bg-gray-200"
                onClick={() => {
                  setTool('add');
                  // Add object at center
                  addObjectToScene(type, { x: 512, y: 512 });
                }}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Asset Upload */}
        <div className="mb-6">
          <h4 className="font-medium mb-2">Upload Asset</h4>
          <input
            type="file"
            accept="image/*"
            onChange={handleAssetUpload}
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Multi-view Generation */}
        <div className="mb-6">
          <h4 className="font-medium mb-2">Multi-view</h4>
          <button
            className="w-full p-2 bg-green-500 text-white rounded hover:bg-green-600"
            onClick={() => {
              const angles: ViewAngle[] = [
                { name: 'Front', rotation: 0, elevation: 0 },
                { name: '3/4', rotation: 45, elevation: 0 },
                { name: 'Side', rotation: 90, elevation: 0 },
                { name: 'Back', rotation: 180, elevation: 0 }
              ];
              generateMultiView(angles);
            }}
          >
            Generate Multi-view
          </button>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 flex flex-col">
        {/* Canvas */}
        <div className="flex-1 p-4">
          <canvas
            ref={canvasRef}
            width={scene.canvasSize.x}
            height={scene.canvasSize.y}
            className="border border-gray-300 bg-white cursor-crosshair"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
          />
        </div>

        {/* Status Bar */}
        <div className="bg-gray-200 p-2 text-sm">
          Objects: {scene.objects.length} | 
          Selected: {selectedObject || 'None'} | 
          Tool: {tool}
        </div>
      </div>

      {/* Layers Panel */}
      {showLayers && (
        <div className="w-64 bg-white border-l p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Layers</h3>
            <button
              onClick={() => setShowLayers(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              ×
            </button>
          </div>
          
          <div className="space-y-2">
            {scene.objects.map(obj => (
              <div
                key={obj.id}
                className={`p-2 rounded cursor-pointer ${
                  selectedObject === obj.id ? 'bg-blue-100' : 'bg-gray-100'
                }`}
                onClick={() => setSelectedObject(obj.id)}
              >
                <div className="flex justify-between items-center">
                  <span className="text-sm">{obj.type}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeObjectFromScene(obj.id);
                    }}
                    className="text-red-500 hover:text-red-700"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Toggle Layers Button */}
      {!showLayers && (
        <button
          onClick={() => setShowLayers(true)}
          className="absolute top-4 right-4 bg-white p-2 rounded shadow"
        >
          Layers
        </button>
      )}
    </div>
  );
}
