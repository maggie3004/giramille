"use client";

import React, { useRef, useState } from 'react';

export interface AssetUploaderProps {
  onAssetUpload: (file: File, type: 'image' | 'vector') => void;
  onAssetInsert: (assetId: string, position: { x: number; y: number }) => void;
  className?: string;
}

export default function AssetUploader({ onAssetUpload, onAssetInsert, className = '' }: AssetUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedAssets, setUploadedAssets] = useState<Array<{ id: string; name: string; src: string; type: 'image' | 'vector' }>>([]);

  const handleFileSelect = (files: FileList | null) => {
    if (!files) return;

    Array.from(files).forEach(file => {
      const type = file.type.startsWith('image/') ? 'image' : 'vector';
      onAssetUpload(file, type);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const src = e.target?.result as string;
        const asset = {
          id: Math.random().toString(36).substr(2, 9),
          name: file.name,
          src,
          type: type as 'image' | 'vector'
        };
        setUploadedAssets(prev => [...prev, asset]);
      };
      reader.readAsDataURL(file);
    });
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleAssetDragStart = (e: React.DragEvent, asset: any) => {
    e.dataTransfer.setData('application/json', JSON.stringify(asset));
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div className={`w-full h-full ${className}`}>
      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
          isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="text-gray-500">
          <svg className="mx-auto h-12 w-12 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p className="text-sm">
            Drag & drop assets here or click to upload
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Supports images (PNG, JPG) and vectors (SVG, AI)
          </p>
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,.svg,.ai"
        className="hidden"
        onChange={(e) => handleFileSelect(e.target.files)}
      />

      {/* Uploaded Assets Grid */}
      {uploadedAssets.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Uploaded Assets</h4>
          <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
            {uploadedAssets.map(asset => (
              <div
                key={asset.id}
                className="relative group cursor-move border border-gray-200 rounded p-2 hover:border-blue-300 transition-colors"
                draggable
                onDragStart={(e) => handleAssetDragStart(e, asset)}
                title={`Drag to canvas to insert: ${asset.name}`}
              >
                {asset.type === 'image' ? (
                  <img
                    src={asset.src}
                    alt={asset.name}
                    className="w-full h-16 object-cover rounded"
                  />
                ) : (
                  <div className="w-full h-16 bg-gray-100 rounded flex items-center justify-center">
                    <span className="text-xs text-gray-500">📐 {asset.name}</span>
                  </div>
                )}
                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all rounded flex items-center justify-center">
                  <span className="text-white text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                    Drag to insert
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
