"use client";

import React, { useState, useRef } from 'react';

export interface MultiViewGeneratorProps {
  sourceImage: string | null;
  onViewGenerated: (view: string, angle: string) => void;
  className?: string;
}

export default function MultiViewGenerator({ sourceImage, onViewGenerated, className = '' }: MultiViewGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedAngle, setSelectedAngle] = useState<string>('front');
  const [generatedViews, setGeneratedViews] = useState<Record<string, string>>({});

  const angles = [
    { key: 'front', label: 'Front View', icon: '👤' },
    { key: 'back', label: 'Back View', icon: '👤' },
    { key: 'left', label: 'Left Side', icon: '👈' },
    { key: 'right', label: 'Right Side', icon: '👉' },
    { key: 'top', label: 'Top View', icon: '⬆️' },
    { key: 'bottom', label: 'Bottom View', icon: '⬇️' },
    { key: '3quarter', label: '3/4 View', icon: '🔄' },
    { key: 'profile', label: 'Profile', icon: '👁️' }
  ];

  const generateView = async (angle: string) => {
    if (!sourceImage) return;

    setIsGenerating(true);
    try {
      // Simulate AI generation - in production, this would call your backend
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate a mock result (in production, this would be the actual AI output)
      const mockResult = generateMockView(sourceImage, angle);
      
      setGeneratedViews(prev => ({ ...prev, [angle]: mockResult }));
      onViewGenerated(mockResult, angle);
    } catch (error) {
      console.error('Error generating view:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const generateMockView = (src: string, angle: string): string => {
    // Create a mock transformed image based on the angle
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    
    // Different transformations for different angles
    switch (angle) {
      case 'front':
        ctx.fillStyle = '#4ecdc4';
        break;
      case 'back':
        ctx.fillStyle = '#45b7d1';
        break;
      case 'left':
        ctx.fillStyle = '#96ceb4';
        break;
      case 'right':
        ctx.fillStyle = '#feca57';
        break;
      case 'top':
        ctx.fillStyle = '#ff9ff3';
        break;
      case 'bottom':
        ctx.fillStyle = '#54a0ff';
        break;
      case '3quarter':
        ctx.fillStyle = '#5f27cd';
        break;
      case 'profile':
        ctx.fillStyle = '#00d2d3';
        break;
      default:
        ctx.fillStyle = '#ff6b6b';
    }
    
    ctx.fillRect(0, 0, 512, 512);
    ctx.fillStyle = '#fff';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`${angle.toUpperCase()} VIEW`, 256, 256);
    
    return canvas.toDataURL('image/png');
  };

  const handleAngleSelect = (angle: string) => {
    setSelectedAngle(angle);
    if (!generatedViews[angle]) {
      generateView(angle);
    }
  };

  return (
    <div className={`w-full h-full ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Multi-View Generation</h3>
        <p className="text-sm text-gray-600">
          Generate the same object from different angles while preserving style
        </p>
      </div>

      {/* Source Image */}
      {sourceImage && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Source Image</h4>
          <div className="w-full h-32 bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={sourceImage}
              alt="Source"
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      )}

      {/* Angle Selection */}
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Select View Angle</h4>
        <div className="grid grid-cols-2 gap-2">
          {angles.map(angle => (
            <button
              key={angle.key}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedAngle === angle.key
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 hover:border-gray-300 text-gray-700'
              }`}
              onClick={() => handleAngleSelect(angle.key)}
              disabled={isGenerating}
            >
              <div className="text-2xl mb-1">{angle.icon}</div>
              <div className="text-xs font-medium">{angle.label}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Generation Status */}
      {isGenerating && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
            <span className="text-sm text-blue-700">
              Generating {selectedAngle} view...
            </span>
          </div>
        </div>
      )}

      {/* Generated Views */}
      {Object.keys(generatedViews).length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Generated Views</h4>
          <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
            {Object.entries(generatedViews).map(([angle, view]) => (
              <div key={angle} className="relative group">
                <img
                  src={view}
                  alt={`${angle} view`}
                  className="w-full h-20 object-cover rounded border border-gray-200"
                />
                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all rounded flex items-center justify-center">
                  <span className="text-white text-xs opacity-0 group-hover:opacity-100 transition-opacity font-medium">
                    {angle.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2">
        <button
          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          onClick={() => generateView(selectedAngle)}
          disabled={isGenerating || !sourceImage}
        >
          Generate {selectedAngle} View
        </button>
        <button
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          onClick={() => setGeneratedViews({})}
        >
          Clear All
        </button>
      </div>
    </div>
  );
}
