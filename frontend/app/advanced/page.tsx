'use client';

import React, { useState, useRef, useCallback } from 'react';

// Advanced AI Image Generation System
export default function AdvancedPage() {
  const [currentMode, setCurrentMode] = useState<'generate' | 'edit' | 'multi-view' | 'vector'>('generate');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [generatePrompt, setGeneratePrompt] = useState('');
  const [generateStyle, setGenerateStyle] = useState('giramille');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Mode-specific handlers
  const handleModeChange = (mode: 'generate' | 'edit' | 'multi-view' | 'vector') => {
    setCurrentMode(mode);
  };

  const handleGenerateImage = async (prompt: string, style: string) => {
    setIsGenerating(true);
    
    try {
      console.log('🔄 Calling backend API for advanced generation...');
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt,
          style: style,
          quality: 'high',
          width: 512,
          height: 512
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.image) {
        console.log('✅ AI Generated image from backend:', data.image.length);
        setGeneratedImages(prev => [data.image, ...prev]);
        setSelectedImage(data.image);
      } else {
        throw new Error(data.error || 'No image generated');
      }
      
    } catch (error) {
      console.error('Generation error:', error);
      alert('Error generating image: ' + (error as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleMultiViewGeneration = async (imagePath: string) => {
    setIsGenerating(true);
    
    try {
      console.log('🔄 Calling backend API for multi-view generation...');
      const response = await fetch('/api/multiview/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imagePath,
          views: ['front', '3/4_left', 'side', 'back'],
          quality: 'high'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.views) {
        console.log('✅ Multi-view generated from backend:', data.views.length, 'views');
        setGeneratedImages(prev => [...data.views, ...prev]);
      } else {
        throw new Error(data.error || 'Multi-view generation failed');
      }
      
    } catch (error) {
      console.error('Multi-view generation error:', error);
      alert('Error generating multi-view: ' + (error as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleVectorConversion = async (imagePath: string) => {
    setIsGenerating(true);
    
    try {
      console.log('🔄 Calling backend API for vector conversion...');
      const response = await fetch('/api/vectorize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imagePath
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // For SVG response, convert to data URL for preview
      const svgContent = await response.text();
      const vectorDataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgContent)}`;
      
      console.log('✅ Vector conversion completed');
      setGeneratedImages(prev => [vectorDataUrl, ...prev]);
      setSelectedImage(vectorDataUrl);
      
    } catch (error) {
      console.error('Vector conversion error:', error);
      alert('Error converting to vector: ' + (error as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">Giramille AI Studio</h1>
              <span className="ml-3 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                Advanced
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => window.location.href = '/'}
                className="text-gray-500 hover:text-gray-700"
              >
                ← Back to Basic
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mode Selector */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'generate', label: 'AI Generation', icon: '🎨' },
              { id: 'edit', label: 'Scene Editor', icon: '✏️' },
              { id: 'multi-view', label: 'Multi-View', icon: '🔄' },
              { id: 'vector', label: 'Vector Export', icon: '📐' }
            ].map(mode => (
              <button
                key={mode.id}
                onClick={() => handleModeChange(mode.id as any)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm ${
                  currentMode === mode.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="text-lg">{mode.icon}</span>
                <span>{mode.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentMode === 'generate' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">AI Image Generation</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Prompt
                  </label>
                  <textarea
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    rows={4}
                    placeholder="Describe the image you want to generate..."
                    value={generatePrompt}
                    onChange={(e) => setGeneratePrompt(e.target.value)}
                  />
                  
                  <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">
                    Style
                  </label>
                  <select 
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    value={generateStyle}
                    onChange={(e) => setGenerateStyle(e.target.value)}
                  >
                    <option value="giramille">Giramille Style</option>
                    <option value="realistic">Realistic</option>
                    <option value="cartoon">Cartoon</option>
                    <option value="abstract">Abstract</option>
                  </select>
                  
                  <button
                    onClick={() => handleGenerateImage(generatePrompt, generateStyle)}
                    disabled={isGenerating || !generatePrompt.trim()}
                    className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50"
                  >
                    {isGenerating ? 'Generating...' : 'Generate Image'}
                  </button>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Generated Images</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {generatedImages.slice(0, 4).map((img, idx) => (
                      <img
                        key={idx}
                        src={img}
                        alt={`Generated ${idx + 1}`}
                        className="w-full h-32 object-cover rounded cursor-pointer hover:opacity-75"
                        onClick={() => setSelectedImage(img)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentMode === 'edit' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Advanced Scene Editor</h2>
              <p className="text-gray-600 mb-6">
                Freepik-level modular editing with real-time manipulation
              </p>
              
              <div className="p-8 text-center text-gray-500">
                <h3 className="text-xl font-semibold mb-4">Advanced Scene Editor</h3>
                <p>Scene editing functionality will be available soon.</p>
              </div>
            </div>
          </div>
        )}

        {currentMode === 'multi-view' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Multi-View Generation</h2>
              <p className="text-gray-600 mb-6">
                Adobe Illustrator-style multi-view generation from different angles
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Image
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    onChange={(e) => setUploadedFile(e.target.files?.[0] || null)}
                  />
                  
                  <button
                    onClick={() => uploadedFile && handleMultiViewGeneration(URL.createObjectURL(uploadedFile))}
                    disabled={isGenerating || !uploadedFile}
                    className="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:opacity-50"
                  >
                    {isGenerating ? 'Generating Views...' : 'Generate Multi-View'}
                  </button>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Generated Views</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {generatedImages.slice(0, 4).map((img, idx) => (
                      <div key={idx} className="text-center">
                        <img
                          src={img}
                          alt={`View ${idx + 1}`}
                          className="w-full h-24 object-cover rounded cursor-pointer hover:opacity-75"
                        />
                        <p className="text-xs text-gray-500 mt-1">
                          {['Front', '3/4 Left', 'Side', 'Back'][idx] || `View ${idx + 1}`}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentMode === 'vector' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Vector Export</h2>
              <p className="text-gray-600 mb-6">
                Professional PNG→Vector conversion with layer separation
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload PNG Image
                  </label>
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    onChange={(e) => setUploadedFile(e.target.files?.[0] || null)}
                  />
                  
                  <div className="mt-4 space-y-2">
                    <label className="block text-sm font-medium text-gray-700">
                      Export Format
                    </label>
                    <div className="flex space-x-4">
                      <label className="flex items-center">
                        <input type="radio" name="format" value="svg" defaultChecked className="mr-2" />
                        SVG
                      </label>
                      <label className="flex items-center">
                        <input type="radio" name="format" value="ai" className="mr-2" />
                        AI
                      </label>
                      <label className="flex items-center">
                        <input type="radio" name="format" value="eps" className="mr-2" />
                        EPS
                      </label>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => uploadedFile && handleVectorConversion(URL.createObjectURL(uploadedFile))}
                    disabled={isGenerating || !uploadedFile}
                    className="mt-4 w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 disabled:opacity-50"
                  >
                    {isGenerating ? 'Converting...' : 'Convert to Vector'}
                  </button>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Vector Preview</h3>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    {selectedImage ? (
                      <img
                        src={selectedImage}
                        alt="Vector preview"
                        className="w-full h-48 object-contain"
                      />
                    ) : (
                      <div className="text-gray-500">
                        <p>No vector preview available</p>
                        <p className="text-sm">Upload an image to convert</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Loading Overlay */}
      {isGenerating && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-sm w-full mx-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <div>
                <p className="text-lg font-medium">Processing...</p>
                <p className="text-sm text-gray-500">This may take a few moments</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
