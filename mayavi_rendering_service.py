#!/usr/bin/env python3
"""
Mayavi 3D Rendering Service
Separate service for handling Mayavi 3D visualizations
Runs on a different port from the main Gradio app
"""

import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import base64
import tempfile
import json

# Configure VTK for offscreen rendering
os.environ['VTK_USE_OFFSCREEN'] = '1'
os.environ['DISPLAY'] = ':99'

try:
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)

    from mayavi import mlab
    from scipy import ndimage
    mlab.options.offscreen = True

    MAYAVI_AVAILABLE = True
    print("✅ Mayavi rendering service initialized successfully")

except Exception as e:
    print(f"❌ Mayavi initialization failed: {e}")
    MAYAVI_AVAILABLE = False

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    return jsonify({
        'service': 'Mayavi 3D Rendering Service',
        'version': '1.0.0',
        'status': 'running',
        'mayavi_available': MAYAVI_AVAILABLE,
        'endpoints': {
            'GET /': 'Service information',
            'GET /health': 'Health check',
            'GET /status': 'Detailed status',
            'POST /render': 'Create 3D visualization'
        },
        'usage': {
            'render_endpoint': 'POST /render',
            'required_fields': ['image_data', 'attention_data', 'predicted_class', 'confidence'],
            'data_format': 'base64 encoded numpy arrays'
        }
    })

def create_mayavi_3d_render(image_data, attention_data, predicted_class, confidence):
    """Create Mayavi 3D visualization"""
    if not MAYAVI_AVAILABLE:
        return None, "Mayavi not available"

    try:
        print(f"🎨 Creating Mayavi 3D render for class {predicted_class} with confidence {confidence:.2f}")

        # Convert base64 data back to numpy arrays
        image = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8).reshape(224, 224, 3)
        attention_map = np.frombuffer(base64.b64decode(attention_data), dtype=np.float32).reshape(224, 224)

        # Normalize attention map
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Create height map
        height_scale = 100
        height_map = attention_normalized * height_scale
        height_map_smooth = ndimage.gaussian_filter(height_map, sigma=1.5)

        # Create coordinate grids
        h, w = image.shape[:2]
        x, y = np.mgrid[0:w:1, 0:h:1]

        # Create Mayavi figure
        fig = mlab.figure(size=(1000, 800), bgcolor=(0.05, 0.05, 0.1))
        mlab.clf()

        # Create 3D surface
        surf = mlab.surf(x, y, height_map_smooth.T,
                        colormap='hot',
                        opacity=0.9,
                        warp_scale='auto')

        # Smart TB region detection
        if predicted_class == 1 and confidence > 0.6:  # TB predicted with >60% confidence
            tb_threshold = max(0.7, np.percentile(attention_normalized.astype(np.float32), 90))
            tb_regions = np.where(attention_normalized > tb_threshold)

            if len(tb_regions[0]) > 0:
                sample_size = min(25, len(tb_regions[0]))
                sample_indices = np.random.choice(len(tb_regions[0]), sample_size, replace=False)

                tb_x = tb_regions[1][sample_indices]
                tb_y = tb_regions[0][sample_indices]
                tb_z = height_map_smooth[tb_regions[0][sample_indices], tb_regions[1][sample_indices]] + 8

                # Add TB markers as red spheres
                mlab.points3d(tb_x, tb_y, tb_z,
                            scale_factor=12,
                            color=(1, 0, 0),
                            opacity=0.9,
                            resolution=16)

        elif predicted_class == 0:  # Normal case
            exceptional_threshold = 0.8
            exceptional_regions = np.where(attention_normalized > exceptional_threshold)

            if len(exceptional_regions[0]) > 5:
                sample_size = min(10, len(exceptional_regions[0]))
                sample_indices = np.random.choice(len(exceptional_regions[0]), sample_size, replace=False)

                exc_x = exceptional_regions[1][sample_indices]
                exc_y = exceptional_regions[0][sample_indices]
                exc_z = height_map_smooth[exceptional_regions[0][sample_indices], exceptional_regions[1][sample_indices]] + 8

                # Add attention markers as orange spheres
                mlab.points3d(exc_x, exc_y, exc_z,
                            scale_factor=8,
                            color=(1, 0.5, 0),
                            opacity=0.7,
                            resolution=12)

        # Customize view and lighting
        mlab.view(azimuth=45, elevation=65, distance='auto')

        # Add title and labels
        title_text = f"Mayavi 3D TB Analysis - {['Normal', 'TB Positive'][predicted_class]}"
        mlab.title(title_text, size=0.8)

        # Add confidence text
        confidence_text = f"Confidence: {confidence:.1%}"
        mlab.text3d(w*0.05, h*0.9, height_scale + 20, confidence_text,
                   scale=8, color=(1, 1, 1))

        # Add region labels
        if predicted_class == 1 and confidence > 0.6:
            mlab.text3d(w*0.05, h*0.8, height_scale + 15, "Red: Potential TB Regions",
                       scale=6, color=(1, 0, 0))
        elif predicted_class == 0:
            mlab.text3d(w*0.05, h*0.8, height_scale + 15, "Orange: High Attention Areas",
                       scale=6, color=(1, 0.5, 0))

        # Add colorbar
        mlab.colorbar(surf, title="Attention Level", orientation='vertical')

        # Force render
        mlab.draw()

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f'mayavi_render_{np.random.randint(1000, 9999)}.png')

        mlab.savefig(temp_filename, size=(1000, 800))

        # Verify and load image
        if os.path.exists(temp_filename):
            file_size = os.path.getsize(temp_filename)
            if file_size > 1000:
                mayavi_image = Image.open(temp_filename)

                # Convert to base64 for transmission
                img_buffer = io.BytesIO()
                mayavi_image.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

                # Clean up
                mlab.close(fig)
                os.remove(temp_filename)

                print("✅ Mayavi render completed successfully")
                return img_base64, "success"
            else:
                print(f"❌ File too small: {file_size} bytes")
        else:
            print("❌ File not created")

        mlab.close(fig)
        return None, "Rendering failed"

    except Exception as e:
        print(f"❌ Mayavi rendering error: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mayavi_available': MAYAVI_AVAILABLE,
        'service': 'Mayavi 3D Rendering Service'
    })

@app.route('/render', methods=['POST'])
def render_3d():
    """Main rendering endpoint"""
    try:
        data = request.json

        # Extract data
        image_data = data['image_data']
        attention_data = data['attention_data']
        predicted_class = data['predicted_class']
        confidence = data['confidence']

        # Create Mayavi render
        img_base64, status = create_mayavi_3d_render(
            image_data, attention_data, predicted_class, confidence
        )

        if img_base64:
            return jsonify({
                'success': True,
                'image_data': img_base64,
                'message': 'Rendering completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': status,
                'message': 'Rendering failed'
            }), 500

    except Exception as e:
        print(f"❌ Render endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Request processing failed'
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get service status"""
    return jsonify({
        'mayavi_available': MAYAVI_AVAILABLE,
        'vtk_configured': 'VTK_USE_OFFSCREEN' in os.environ,
        'service_name': 'Mayavi 3D Rendering Service',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Mayavi 3D Rendering Service...")
    print(f"🔬 Mayavi Available: {MAYAVI_AVAILABLE}")
    print("🌐 Service will run on http://localhost:5001")
    print("📡 Endpoints:")
    print("   - GET  /health  - Health check")
    print("   - POST /render  - 3D rendering")
    print("   - GET  /status  - Service status")

    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
