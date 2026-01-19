import React, { useState } from 'react';
import axios from 'axios';
import ModelViewer3D from '../../components/ModelViewer3D';

/**
 * 3D Generation Page
 * Generate high-precision 3D models from text prompts
 */
const Generate3D = () => {
  const [prompt, setPrompt] = useState('');
  const [quality, setQuality] = useState('very_high');
  const [includePreview, setIncludePreview] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [generationTime, setGenerationTime] = useState(null);

  const handleGenerate = async (e) => {
    e.preventDefault();

    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const startTime = Date.now();

      const response = await axios.post('http://localhost:5000/api/generate/3d', {
        prompt: prompt.trim(),
        quality,
        include_preview: includePreview,
        steps_2d: quality === 'very_high' ? 50 : quality === 'high' ? 40 : 30,
        steps_3d: quality === 'very_high' ? 64 : quality === 'high' ? 48 : 32,
      });

      const elapsed = (Date.now() - startTime) / 1000;
      setGenerationTime(elapsed);

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError(response.data.error || 'Generation failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message);
      console.error('Generation error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="generate-3d-page" style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
      <style>{`
        .generate-3d-page {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
          padding: 20px;
        }

        .container-3d {
          background: white;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }

        .generation-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          height: 100%;
        }

        .generation-form-section {
          padding: 30px;
          overflow-y: auto;
          background: #f8f9fa;
        }

        .generation-preview-section {
          padding: 30px;
          display: flex;
          flex-direction: column;
          gap: 20px;
          overflow-y: auto;
        }

        .form-group {
          margin-bottom: 20px;
        }

        .form-group label {
          display: block;
          margin-bottom: 8px;
          font-weight: 600;
          color: #333;
          font-size: 14px;
        }

        .form-group input,
        .form-group textarea,
        .form-group select {
          width: 100%;
          padding: 12px;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-size: 14px;
          font-family: inherit;
        }

        .form-group textarea {
          resize: vertical;
          min-height: 80px;
        }

        .form-group input:focus,
        .form-group textarea:focus,
        .form-group select:focus {
          outline: none;
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .quality-options {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }

        .quality-option {
          padding: 10px;
          border: 2px solid #ddd;
          border-radius: 6px;
          cursor: pointer;
          text-align: center;
          transition: all 0.3s ease;
        }

        .quality-option:hover {
          border-color: #667eea;
          background: #f0f4ff;
        }

        .quality-option input {
          margin-right: 5px;
        }

        .quality-option.selected {
          border-color: #667eea;
          background: #667eea;
          color: white;
        }

        .checkbox-group {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .checkbox-group input[type='checkbox'] {
          width: auto;
          margin: 0;
        }

        .generate-btn {
          width: 100%;
          padding: 14px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 6px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .generate-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .generate-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .loading-spinner {
          display: inline-block;
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-top: 3px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-right: 10px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .preview-title {
          font-size: 18px;
          font-weight: 600;
          color: #333;
          margin-bottom: 15px;
        }

        .preview-image {
          width: 100%;
          height: 350px;
          border-radius: 8px;
          object-fit: contain;
          background: #f0f0f0;
        }

        .model-viewer-container {
          width: 100%;
          height: 400px;
          border-radius: 8px;
          overflow: hidden;
          border: 1px solid #ddd;
        }

        .info-box {
          background: #e8f4f8;
          border-left: 4px solid #667eea;
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 20px;
          font-size: 14px;
          color: #333;
        }

        .success-box {
          background: #e8f8f0;
          border-left: 4px solid #00a86b;
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 20px;
          font-size: 14px;
          color: #333;
        }

        .error-box {
          background: #fde8e8;
          border-left: 4px solid #d32f2f;
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 20px;
          font-size: 14px;
          color: #d32f2f;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          margin-top: 15px;
        }

        .stat-item {
          background: #f5f5f5;
          padding: 10px;
          border-radius: 6px;
          font-size: 12px;
        }

        .stat-label {
          color: #999;
          margin-bottom: 3px;
        }

        .stat-value {
          font-weight: 600;
          color: #333;
          font-size: 14px;
        }

        .download-btn {
          padding: 10px 20px;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: background 0.2s ease;
        }

        .download-btn:hover {
          background: #764ba2;
        }

        @media (max-width: 1024px) {
          .generation-grid {
            grid-template-columns: 1fr;
          }

          .preview-image {
            height: 250px;
          }
        }
      `}</style>

      <h1 style={{ color: 'white', marginBottom: '20px', textAlign: 'center', fontSize: '32px' }}>
        🎨 3D Model Generation Studio
      </h1>

      <div className="container-3d">
        <div className="generation-grid">
          {/* Form Section */}
          <div className="generation-form-section">
            <h2 style={{ marginBottom: '20px', color: '#333' }}>Generate 3D Model</h2>

            <div className="info-box">
              <strong>⚡ High-Precision Generation</strong>
              <p>Directly generates 3D models from text prompts with maximum accuracy. Includes 2D preview and quality validation.</p>
            </div>

            <form onSubmit={handleGenerate}>
              <div className="form-group">
                <label>📝 What would you like to generate?</label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g., a black horse with detailed features, glossy coat, professional sculpture..."
                  disabled={loading}
                />
              </div>

              <div className="form-group">
                <label>✨ Quality Level</label>
                <div className="quality-options">
                  <label
                    className={`quality-option ${quality === 'high' ? 'selected' : ''}`}
                    style={{ cursor: loading ? 'not-allowed' : 'pointer' }}
                  >
                    <input
                      type="radio"
                      name="quality"
                      value="high"
                      checked={quality === 'high'}
                      onChange={() => setQuality('high')}
                      disabled={loading}
                    />
                    High (Faster)
                  </label>
                  <label
                    className={`quality-option ${quality === 'very_high' ? 'selected' : ''}`}
                    style={{ cursor: loading ? 'not-allowed' : 'pointer' }}
                  >
                    <input
                      type="radio"
                      name="quality"
                      value="very_high"
                      checked={quality === 'very_high'}
                      onChange={() => setQuality('very_high')}
                      disabled={loading}
                    />
                    Very High (Best)
                  </label>
                </div>
              </div>

              <div className="form-group">
                <label className="checkbox-group">
                  <input
                    type="checkbox"
                    checked={includePreview}
                    onChange={(e) => setIncludePreview(e.target.checked)}
                    disabled={loading}
                  />
                  <span>Include 2D preview image</span>
                </label>
              </div>

              <button
                type="submit"
                className="generate-btn"
                disabled={loading || !prompt.trim()}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Generating 3D Model...
                  </>
                ) : (
                  '🚀 Generate 3D Model'
                )}
              </button>

              {generationTime && (
                <div style={{ marginTop: '15px', fontSize: '12px', color: '#666' }}>
                  ⏱️ Last generation took {generationTime.toFixed(1)} seconds
                </div>
              )}
            </form>
          </div>

          {/* Preview Section */}
          <div className="generation-preview-section">
            <h2 style={{ color: '#333' }}>Preview & Results</h2>

            {error && (
              <div className="error-box">
                <strong>❌ Error</strong>
                <p>{error}</p>
              </div>
            )}

            {loading && (
              <div className="info-box">
                <strong>🔄 Generating...</strong>
                <p>This may take 2-5 minutes depending on quality settings.</p>
                <ul style={{ marginTop: '10px' }}>
                  <li>Step 1: Optimizing prompt</li>
                  <li>Step 2: Generating high-precision 2D preview</li>
                  <li>Step 3: Generating 3D model (native generation)</li>
                  <li>Step 4: Validating and optimizing mesh</li>
                </ul>
              </div>
            )}

            {result && !loading && (
              <div className="success-box">
                <strong>✅ Generation Complete!</strong>
                <p>Your 3D model has been generated successfully.</p>

                {result.preview_2d_path && includePreview && (
                  <div style={{ marginTop: '15px' }}>
                    <p style={{ marginBottom: '10px', fontWeight: '600' }}>2D Preview:</p>
                    <img src={result.preview_2d_path} alt="2D Preview" className="preview-image" />
                  </div>
                )}

                {result.model_formats?.obj && (
                  <div style={{ marginTop: '20px' }}>
                    <p style={{ marginBottom: '10px', fontWeight: '600' }}>3D Model Viewer:</p>
                    <ModelViewer3D
                      modelPath={result.model_formats.obj}
                      format="obj"
                      title={result.prompt}
                    />
                  </div>
                )}

                <div className="stats-grid">
                  <div className="stat-item">
                    <div className="stat-label">Model ID</div>
                    <div className="stat-value" style={{ fontSize: '12px', wordBreak: 'break-all' }}>
                      {result.model_id}
                    </div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-label">Format</div>
                    <div className="stat-value">OBJ + PLY</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-label">Vertices</div>
                    <div className="stat-value">
                      {result.validation?.vertices?.toLocaleString() || 'N/A'}
                    </div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-label">Faces</div>
                    <div className="stat-value">
                      {result.validation?.faces?.toLocaleString() || 'N/A'}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                  {result.model_formats?.obj && (
                    <a
                      href={`http://localhost:5000/api/model/${result.model_id}/download?format=obj`}
                      className="download-btn"
                      download
                    >
                      📥 Download OBJ
                    </a>
                  )}
                  {result.model_formats?.ply && (
                    <a
                      href={`http://localhost:5000/api/model/${result.model_id}/download?format=ply`}
                      className="download-btn"
                      download
                    >
                      📥 Download PLY
                    </a>
                  )}
                </div>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="info-box">
                <strong>ℹ️ How it works:</strong>
                <ol style={{ marginTop: '10px', paddingLeft: '20px' }}>
                  <li>Enter a detailed description of what you want to create</li>
                  <li>Choose your preferred quality level</li>
                  <li>Click "Generate 3D Model"</li>
                  <li>Watch as we generate high-precision 3D models</li>
                  <li>Download in OBJ or PLY format</li>
                </ol>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Generate3D;
