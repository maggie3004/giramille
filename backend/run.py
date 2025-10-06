#!/usr/bin/env python3
"""
Backend server for the AI Image Generation and Vectorization Pipeline
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"Starting backend server on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Upload folder: {os.path.abspath('uploads')}")
    print(f"Output folder: {os.path.abspath('outputs')}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
