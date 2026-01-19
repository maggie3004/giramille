import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const ModelViewer3D = ({ modelPath, format = 'obj', title = '3D Model' }) => {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!modelPath || !containerRef.current) return;

    const initScene = async () => {
      try {
        setLoading(true);
        setError(null);

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        sceneRef.current = scene;

        // Camera setup
        const camera = new THREE.PerspectiveCamera(
          75,
          containerRef.current.clientWidth / containerRef.current.clientHeight,
          0.1,
          1000
        );
        camera.position.set(0, 2, 5);
        cameraRef.current = camera;

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(
          containerRef.current.clientWidth,
          containerRef.current.clientHeight
        );
        renderer.shadowMap.enabled = true;
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);

        // Load model
        let model = null;
        if (format === 'obj') {
          const loader = new OBJLoader();
          const text = await fetch(modelPath).then(r => r.text());
          model = loader.parse(text);
        } else if (format === 'gltf' || format === 'glb') {
          const loader = new GLTFLoader();
          const gltf = await loader.loadAsync(modelPath);
          model = gltf.scene;
        }

        if (model) {
          // Center and scale model
          const box = new THREE.Box3().setFromObject(model);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 2 / maxDim;

          model.position.sub(center);
          model.scale.multiplyScalar(scale);
          scene.add(model);

          // Set camera to view model
          const distance = maxDim * 1.5;
          camera.position.z = distance;
          camera.lookAt(0, 0, 0);
        }

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true;
        controls.autoRotateSpeed = 4;
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controlsRef.current = controls;

        // Handle window resize
        const handleResize = () => {
          if (!containerRef.current) return;
          const width = containerRef.current.clientWidth;
          const height = containerRef.current.clientHeight;
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        };
        window.addEventListener('resize', handleResize);

        // Animation loop
        const animate = () => {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        };
        animate();

        setLoading(false);

      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    initScene();

    // Cleanup function
    return () => {
      if (rendererRef.current && containerRef.current) {
        try {
          containerRef.current.removeChild(rendererRef.current.domElement);
        } catch (e) {
          // Already removed
        }
        rendererRef.current.dispose();
      }
    };
  }, [modelPath, format]);

  return (
    <div className="model-viewer-container" style={{ width: '100%', height: '100%' }}>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
        }}
      >
        {loading && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: 'white',
              fontSize: '18px',
              zIndex: 10,
            }}
          >
            Loading 3D Model...
          </div>
        )}
        {error && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: 'red',
              fontSize: '16px',
              zIndex: 10,
            }}
          >
            Error loading model: {error}
          </div>
        )}
      </div>
      <div
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          color: 'white',
          fontSize: '12px',
          backgroundColor: 'rgba(0,0,0,0.5)',
          padding: '8px 12px',
          borderRadius: '4px',
          zIndex: 5,
        }}
      >
        <p style={{ margin: '0 0 5px 0' }}>
          <strong>{title}</strong>
        </p>
        <p style={{ margin: '0' }}>
          Right-click to rotate | Scroll to zoom | Middle-click to pan
        </p>
      </div>
    </div>
  );
};

export default ModelViewer3D;
