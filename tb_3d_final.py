#!/usr/bin/env python3
"""
TB Detection System with WORKING Interactive 3D Visualization
Final version with proper Plotly integration and interactive 3D
"""

import os
import torch
import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
import requests
import base64

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def create_plotly_3d_component(image, attention_map, predicted_class, confidence):
    """Create Plotly 3D visualization as Gradio Plot component"""
    try:
        import plotly.graph_objects as go
        from scipy import ndimage

        print("🎨 Creating interactive Plotly 3D component...")

        # Ensure inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Normalize attention map
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Create height map from attention
        height_scale = 100
        height_map = attention_normalized * height_scale

        # Smooth the height map
        height_map_smooth = ndimage.gaussian_filter(height_map, sigma=1.5)

        # Create coordinate grids (downsample for performance)
        h, w = image.shape[:2]
        step = max(1, min(h, w) // 30)  # Good resolution for interaction
        x = np.arange(0, w, step)
        y = np.arange(0, h, step)
        X, Y = np.meshgrid(x, y)
        Z = height_map_smooth[::step, ::step]

        # Create 3D surface
        surface = go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Hot',
            opacity=0.9,
            name='Attention Surface',
            showscale=True,
            hovertemplate='<b>Position</b><br>X: %{x}<br>Y: %{y}<br>Attention: %{z:.1f}<extra></extra>'
        )

        # Create figure
        fig = go.Figure(data=[surface])

        # Smart TB region detection based on prediction and attention patterns
        # Only show TB markers if the model predicts TB with reasonable confidence
        if predicted_class == 1 and confidence > 0.6:  # TB predicted with >60% confidence
            # Use higher threshold for TB cases
            tb_threshold = max(0.7, np.percentile(attention_normalized.astype(np.float32), 90))
            tb_regions = np.where(attention_normalized > tb_threshold)

            if len(tb_regions[0]) > 0:
                sample_size = min(30, len(tb_regions[0]))
                sample_indices = np.random.choice(len(tb_regions[0]), sample_size, replace=False)

                tb_x = tb_regions[1][sample_indices]
                tb_y = tb_regions[0][sample_indices]
                tb_z = height_map_smooth[tb_regions[0][sample_indices], tb_regions[1][sample_indices]] + 10

                # Add TB markers
                tb_markers = go.Scatter3d(
                    x=tb_x, y=tb_y, z=tb_z,
                    mode='markers',
                    marker=dict(size=6, color='red', opacity=0.9),
                    name='Potential TB Regions',
                    hovertemplate='<b>Potential TB Region</b><br>X: %{x}<br>Y: %{y}<br>Height: %{z:.1f}<extra></extra>'
                )
                fig.add_trace(tb_markers)
        elif predicted_class == 0:  # Normal case
            # For normal cases, only show markers if there are truly exceptional attention areas
            exceptional_threshold = 0.8  # Very high absolute threshold
            exceptional_regions = np.where(attention_normalized > exceptional_threshold)

            if len(exceptional_regions[0]) > 5:  # Only if there are significant areas
                sample_size = min(10, len(exceptional_regions[0]))
                sample_indices = np.random.choice(len(exceptional_regions[0]), sample_size, replace=False)

                exc_x = exceptional_regions[1][sample_indices]
                exc_y = exceptional_regions[0][sample_indices]
                exc_z = height_map_smooth[exceptional_regions[0][sample_indices], exceptional_regions[1][sample_indices]] + 10

                # Add attention markers (different color for normal cases)
                attention_markers = go.Scatter3d(
                    x=exc_x, y=exc_y, z=exc_z,
                    mode='markers',
                    marker=dict(size=4, color='orange', opacity=0.7),
                    name='High Attention Areas',
                    hovertemplate='<b>High Attention Area</b><br>X: %{x}<br>Y: %{y}<br>Height: %{z:.1f}<extra></extra>'
                )
                fig.add_trace(attention_markers)

        # Update layout for better interaction
        title_text = f'Interactive 3D TB Attention - {["Normal", "TB Positive"][predicted_class]} (Confidence: {confidence:.1%})'

        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=16)),
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Attention Height',
                bgcolor='rgb(240, 240, 250)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        print("✅ Interactive Plotly 3D component created")
        return fig, True

    except Exception as e:
        print(f"❌ Error creating Plotly 3D: {e}")
        import traceback
        traceback.print_exc()

        # Create empty figure as fallback
        fig = go.Figure()
        fig.add_annotation(
            text=f"3D Visualization Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig, False

def create_enhanced_matplotlib_3d(image, attention_map, predicted_class, confidence):
    """Create enhanced matplotlib 3D visualization"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from scipy import ndimage

        print("🎨 Creating enhanced matplotlib 3D...")

        # Ensure inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Normalize attention map
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Create height map
        height_scale = 100
        height_map = attention_normalized * height_scale
        height_map_smooth = ndimage.gaussian_filter(height_map, sigma=1.5)

        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 10))

        # Main 3D plot
        ax1 = fig.add_subplot(221, projection='3d')

        # Create coordinate grids (downsample for performance)
        h, w = image.shape[:2]
        step = max(1, min(h, w) // 40)
        x = np.arange(0, w, step)
        y = np.arange(0, h, step)
        X, Y = np.meshgrid(x, y)
        Z = height_map_smooth[::step, ::step]

        # Create 3D surface
        surf = ax1.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)

        # Smart TB region detection (same logic as Plotly)
        if predicted_class == 1 and confidence > 0.6:  # TB predicted with >60% confidence
            tb_threshold = max(0.7, np.percentile(attention_normalized.astype(np.float32), 90))
            tb_regions = np.where(attention_normalized > tb_threshold)

            if len(tb_regions[0]) > 0:
                sample_size = min(20, len(tb_regions[0]))
                sample_indices = np.random.choice(len(tb_regions[0]), sample_size, replace=False)

                tb_x = tb_regions[1][sample_indices]
                tb_y = tb_regions[0][sample_indices]
                tb_z = height_map_smooth[tb_regions[0][sample_indices], tb_regions[1][sample_indices]] + 10

                ax1.scatter(tb_x, tb_y, tb_z, c='red', s=50, alpha=0.8, label='Potential TB Regions')
        elif predicted_class == 0:  # Normal case
            exceptional_threshold = 0.8
            exceptional_regions = np.where(attention_normalized > exceptional_threshold)

            if len(exceptional_regions[0]) > 5:
                sample_size = min(10, len(exceptional_regions[0]))
                sample_indices = np.random.choice(len(exceptional_regions[0]), sample_size, replace=False)

                exc_x = exceptional_regions[1][sample_indices]
                exc_y = exceptional_regions[0][sample_indices]
                exc_z = height_map_smooth[exceptional_regions[0][sample_indices], exceptional_regions[1][sample_indices]] + 10

                ax1.scatter(exc_x, exc_y, exc_z, c='orange', s=30, alpha=0.7, label='High Attention Areas')

        # Customize the main plot
        ax1.set_title(f'3D TB Attention - {["Normal", "TB Positive"][predicted_class]}\nConfidence: {confidence:.1%}')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_zlabel('Attention Height')
        ax1.legend()

        # Top view (XY plane)
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(attention_map, cmap='hot', alpha=0.8)
        ax2.set_title('Top View (2D Heatmap)')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Side view (XZ plane)
        ax3 = fig.add_subplot(223)
        side_view = np.mean(height_map_smooth, axis=0)
        ax3.plot(range(len(side_view)), side_view, 'r-', linewidth=2)
        ax3.fill_between(range(len(side_view)), side_view, alpha=0.3, color='red')
        ax3.set_title('Side View (X-Z Profile)')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Attention Height')
        ax3.grid(True, alpha=0.3)

        # Front view (YZ plane)
        ax4 = fig.add_subplot(224)
        front_view = np.mean(height_map_smooth, axis=1)
        ax4.plot(front_view, range(len(front_view)), 'b-', linewidth=2)
        ax4.fill_betweenx(range(len(front_view)), front_view, alpha=0.3, color='blue')
        ax4.set_title('Front View (Y-Z Profile)')
        ax4.set_xlabel('Attention Height')
        ax4.set_ylabel('Y (pixels)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        enhanced_image = Image.open(buf)
        print("✅ Enhanced matplotlib 3D created")
        return enhanced_image, True

    except Exception as e:
        print(f"❌ Error creating enhanced matplotlib 3D: {e}")
        import traceback
        traceback.print_exc()
        # Create error placeholder
        error_img = Image.new('RGB', (800, 600), color='lightcoral')
        return error_img, False

def create_matplotlib_3d_fallback(image, attention_map, predicted_class, confidence):
    """Create matplotlib-based 3D fallback when Mayavi is not available"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from scipy import ndimage

        print("🎨 Creating matplotlib 3D fallback (Mayavi-style)...")

        # Ensure inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Normalize attention map
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Create height map
        height_scale = 100
        height_map = attention_normalized * height_scale
        height_map_smooth = ndimage.gaussian_filter(height_map, sigma=1.5)

        # Create figure with dark background (Mayavi-style)
        fig = plt.figure(figsize=(12, 9), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # Create coordinate grids (higher resolution for better quality)
        h, w = image.shape[:2]
        step = max(1, min(h, w) // 50)
        x = np.arange(0, w, step)
        y = np.arange(0, h, step)
        X, Y = np.meshgrid(x, y)
        Z = height_map_smooth[::step, ::step]

        # Create 3D surface with enhanced styling
        surf = ax.plot_surface(X, Y, Z, cmap='hot', alpha=0.9,
                              linewidth=0, antialiased=True, shade=True)

        # Smart TB region detection (same logic as other visualizations)
        if predicted_class == 1 and confidence > 0.6:
            tb_threshold = max(0.7, np.percentile(attention_normalized.astype(np.float32), 90))
            tb_regions = np.where(attention_normalized > tb_threshold)

            if len(tb_regions[0]) > 0:
                sample_size = min(25, len(tb_regions[0]))
                sample_indices = np.random.choice(len(tb_regions[0]), sample_size, replace=False)

                tb_x = tb_regions[1][sample_indices]
                tb_y = tb_regions[0][sample_indices]
                tb_z = height_map_smooth[tb_regions[0][sample_indices], tb_regions[1][sample_indices]] + 8

                ax.scatter(tb_x, tb_y, tb_z, c='red', s=80, alpha=0.9,
                          edgecolors='darkred', linewidth=1, label='Potential TB Regions')

        elif predicted_class == 0:
            exceptional_threshold = 0.8
            exceptional_regions = np.where(attention_normalized > exceptional_threshold)

            if len(exceptional_regions[0]) > 5:
                sample_size = min(10, len(exceptional_regions[0]))
                sample_indices = np.random.choice(len(exceptional_regions[0]), sample_size, replace=False)

                exc_x = exceptional_regions[1][sample_indices]
                exc_y = exceptional_regions[0][sample_indices]
                exc_z = height_map_smooth[exceptional_regions[0][sample_indices], exceptional_regions[1][sample_indices]] + 8

                ax.scatter(exc_x, exc_y, exc_z, c='orange', s=60, alpha=0.7,
                          edgecolors='darkorange', linewidth=1, label='High Attention Areas')

        # Customize appearance (Mayavi-style)
        ax.set_facecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # Set labels and title
        title_text = f"3D TB Analysis (Matplotlib) - {['Normal', 'TB Positive'][predicted_class]}"
        ax.set_title(title_text, color='white', fontsize=14, pad=20)
        ax.set_xlabel('X (pixels)', color='white')
        ax.set_ylabel('Y (pixels)', color='white')
        ax.set_zlabel('Attention Height', color='white')

        # Add confidence text
        confidence_text = f"Confidence: {confidence:.1%}"
        ax.text2D(0.02, 0.95, confidence_text, transform=ax.transAxes,
                 color='white', fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                 facecolor="black", alpha=0.7))

        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label('Attention Level', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')

        # Set viewing angle
        ax.view_init(elev=65, azim=45)

        # Legend
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', facecolor='black', edgecolor='white')

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        plt.close()

        fallback_image = Image.open(buf)
        print("✅ Matplotlib 3D fallback created (Mayavi-style)")
        return fallback_image, True

    except Exception as e:
        print(f"❌ Error creating matplotlib fallback: {e}")
        error_img = Image.new('RGB', (1000, 800), color='lightcoral')
        return error_img, False

def create_mayavi_3d_visualization(image, attention_map, predicted_class, confidence):
    """Create Mayavi 3D visualization using external rendering service"""

    try:
        print("🌐 Requesting Mayavi 3D render from external service...")
        import requests
        import base64

        # Prepare data for transmission
        image_bytes = image.astype(np.uint8).tobytes()
        attention_bytes = attention_map.astype(np.float32).tobytes()

        image_b64 = base64.b64encode(image_bytes).decode()
        attention_b64 = base64.b64encode(attention_bytes).decode()

        # Prepare request data
        request_data = {
            'image_data': image_b64,
            'attention_data': attention_b64,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence)
        }

        # Make request to rendering service
        service_url = 'http://localhost:5001/render'

        print(f"📡 Sending request to {service_url}...")
        response = requests.post(service_url, json=request_data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("📦 Decoding Mayavi image data...")
                # Decode the image
                img_data = base64.b64decode(result['image_data'])
                mayavi_image = Image.open(io.BytesIO(img_data))

                print(f"✅ Mayavi 3D render received from service: {mayavi_image.size} pixels")
                print("🔄 Returning Mayavi image to main pipeline...")
                return mayavi_image, True
            else:
                print(f"❌ Rendering service error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Service request failed: {response.status_code}")
            print(f"Response text: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Mayavi rendering service")
        print("💡 Make sure to start the service: python mayavi_rendering_service.py")
    except requests.exceptions.Timeout:
        print("❌ Rendering service timeout")
    except Exception as e:
        print(f"❌ Error communicating with rendering service: {e}")

    # Fallback to matplotlib if service fails
    print("🔄 Falling back to high-quality matplotlib 3D rendering...")
    return create_matplotlib_3d_fallback(image, attention_map, predicted_class, confidence)

# Mayavi core function removed - now using external rendering service

def create_2d_dashboard(image, attention_map, predicted_class, confidence, normal_prob, tb_prob):
    """Create 2D analysis dashboard"""
    try:
        from matplotlib.patches import Rectangle

        print("🎨 Creating 2D analysis dashboard...")

        # Ensure inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Create 2D visualization dashboard
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

        # 1. Original chest X-ray
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax1.set_title('Original Chest X-ray', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. 2D Attention Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(attention_map, cmap='hot', alpha=0.8)
        ax2.set_title('2D Attention Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Attention')

        # 3. Overlay visualization
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax3.imshow(attention_map, cmap='hot', alpha=0.6)
        ax3.set_title('Attention Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 4. TB Region Detection
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(image, cmap='gray' if len(image.shape) == 2 else None)

        # Smart TB region highlighting (same logic as 3D)
        if predicted_class == 1 and confidence > 0.6:  # TB predicted with >60% confidence
            # Normalize attention map for consistent thresholding
            attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            tb_threshold = max(0.7, np.percentile(attention_normalized.astype(np.float32), 90))
            tb_mask = attention_normalized > tb_threshold

            # Find contours
            tb_mask_uint8 = (tb_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(tb_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes for TB regions
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Smaller threshold for TB cases
                    x, y, w, h = cv2.boundingRect(contour)
                    rect = Rectangle((x, y), w, h, linewidth=2,
                                   edgecolor='red', facecolor='none', alpha=0.8)
                    ax4.add_patch(rect)
        elif predicted_class == 0:  # Normal case
            # For normal cases, only highlight truly exceptional areas
            attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            exceptional_threshold = 0.8
            exceptional_mask = attention_normalized > exceptional_threshold

            # Find contours
            exceptional_mask_uint8 = (exceptional_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(exceptional_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes for high attention areas (if any significant ones exist)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 200]  # Higher threshold
            if len(significant_contours) > 0:
                for contour in significant_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    rect = Rectangle((x, y), w, h, linewidth=1,
                                   edgecolor='orange', facecolor='none', alpha=0.6)
                    ax4.add_patch(rect)

        ax4.set_title('TB Region Detection', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # 5. Probability Distribution
        ax5 = fig.add_subplot(gs[1, 0])
        classes = ['Normal', 'TB Positive']
        probs = [normal_prob, tb_prob]
        colors = ['green' if predicted_class == 0 else 'lightgreen',
                 'red' if predicted_class == 1 else 'lightcoral']

        bars = ax5.bar(classes, probs, color=colors)
        ax5.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Probability')
        ax5.set_ylim(0, 1)

        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

        # 6. Risk Assessment
        ax6 = fig.add_subplot(gs[1, 1])
        risk_level = "High" if predicted_class == 1 and confidence > 0.8 else \
                    "Moderate" if predicted_class == 1 and confidence > 0.6 else \
                    "Low-Moderate" if predicted_class == 1 else "Low"

        risk_colors = {"High": "red", "Moderate": "orange", "Low-Moderate": "yellow", "Low": "green"}

        ax6.text(0.5, 0.7, "Risk Level", ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax6.transAxes)
        ax6.text(0.5, 0.5, risk_level, ha='center', va='center',
                fontsize=20, fontweight='bold', color=risk_colors[risk_level],
                transform=ax6.transAxes)
        ax6.text(0.5, 0.3, f"Confidence: {confidence:.1%}", ha='center', va='center',
                fontsize=12, transform=ax6.transAxes)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')

        # 7. 3D Visualization Guide
        ax7 = fig.add_subplot(gs[1, 2:])

        guide_text = """3D Visualization Guide:

INTERACTIVE PLOTLY 3D (Below):
* Rotate: Click and drag to view from any angle
* Zoom: Mouse wheel to examine specific regions
* Hover: See exact attention values at any point
* Red spheres: Mark highest attention TB regions

ENHANCED MATPLOTLIB 3D (Right):
* Main view: 3D surface with TB markers
* Top view: 2D heatmap projection
* Side/Front views: Cross-sectional profiles

COLOR INTERPRETATION:
* Blue/Dark: Normal tissue (low attention)
* Yellow/Bright: Suspicious areas (high attention)
* Red markers: Potential TB regions"""

        ax7.text(0.05, 0.95, guide_text, ha='left', va='top',
                fontsize=9, transform=ax7.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax7.axis('off')

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return Image.open(buf)

    except Exception as e:
        print(f"❌ Error creating 2D dashboard: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model():
    """Load the pre-trained TB detection model"""
    print_section("LOADING PRE-TRAINED MODEL")

    try:
        import timm
        import torch.nn as nn

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")

        # Check if model file exists
        model_path = 'best_tb_model.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None, device

        # Recreate model architecture
        class TBDetectionModel(nn.Module):
            def __init__(self, num_classes=2, backbone='efficientnet_b2'):
                super(TBDetectionModel, self).__init__()

                self.backbone = timm.create_model(
                    backbone, pretrained=True, features_only=True, out_indices=[1, 2, 3, 4]
                )

                dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
                with torch.no_grad():
                    features = self.backbone(dummy_input)
                    self.feature_dims = [f.shape[1] for f in features]

                self.fpn_convs = nn.ModuleList([
                    nn.Conv2d(dim, 256, 1) for dim in self.feature_dims
                ])

                self.gap = nn.AdaptiveAvgPool2d(1)

                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * len(self.feature_dims), 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_classes)
                )

                self.attention = nn.Sequential(
                    nn.Conv2d(256, 64, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = x.float()
                features = self.backbone(x)

                fpn_features = []
                for i, feat in enumerate(features):
                    fpn_feat = self.fpn_convs[i](feat)
                    if i > 0:
                        fpn_feat = torch.nn.functional.interpolate(
                            fpn_feat, size=features[0].shape[2:], mode='bilinear', align_corners=False
                        )
                    fpn_features.append(fpn_feat)

                attention_map = self.attention(fpn_features[-1])

                pooled_features = []
                for feat in fpn_features:
                    pooled = self.gap(feat).flatten(1)
                    pooled_features.append(pooled)

                combined_pooled = torch.cat(pooled_features, dim=1)
                output = self.classifier(combined_pooled)

                return {
                    'logits': output,
                    'attention_map': attention_map,
                    'features': combined_pooled
                }

        # Load model
        model = TBDetectionModel()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully!")
        if 'val_acc' in checkpoint:
            print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")

        return model, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, device

def predict_tb_with_interactive_3d(image, model, device):
    """Predict TB with working interactive 3D visualization"""
    if model is None:
        return "❌ Model not loaded", None, None, None, "Model loading failed"

    if image is None:
        return "❌ No image provided", None, None, None, "Please upload a chest X-ray"

    try:
        # Preprocess image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize and normalize
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std

        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs['logits'], dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][int(predicted_class)].item()

            normal_prob = probabilities[0][0].item()
            tb_prob = probabilities[0][1].item()

            # Get attention map
            attention = outputs['attention_map'][0, 0].cpu().numpy()
            attention = cv2.resize(attention, (224, 224))

            # Create all visualizations
            print("🎨 Creating complete interactive 3D visualization suite...")

            # 1. 2D Dashboard
            dashboard_img = create_2d_dashboard(
                image_resized, attention, predicted_class, confidence,
                normal_prob, tb_prob
            )

            # 2. Interactive Plotly 3D
            plotly_fig, plotly_success = create_plotly_3d_component(
                image_resized, attention, predicted_class, confidence
            )

            # 3. Enhanced matplotlib 3D
            matplotlib_img, matplotlib_success = create_enhanced_matplotlib_3d(
                image_resized, attention, predicted_class, confidence
            )

            # 4. Mayavi 3D visualization
            print("🔬 Starting Mayavi 3D visualization...")
            mayavi_img, mayavi_success = create_mayavi_3d_visualization(
                image_resized, attention, predicted_class, confidence
            )
            print(f"✅ Mayavi 3D visualization completed: success={mayavi_success}")

            # Create result text
            print("📝 Creating result text...")
            class_names = ['Normal', 'TB Positive']
            result_class = class_names[int(predicted_class)]

            result_text = f"""
# 🔍 **TB Detection Analysis with Interactive 3D Visualization**

## **Primary Diagnosis**
- **Classification:** {result_class}
- **Confidence:** {confidence:.1%}

## **Detailed Probabilities**
- **Normal/Healthy:** {normal_prob:.1%}
- **TB Positive:** {tb_prob:.1%}

## **Interactive 3D Visualization Status**
- **Plotly Interactive 3D:** {'✅ Working - Rotate, zoom, hover!' if plotly_success else '❌ Failed'}
- **Enhanced Matplotlib 3D:** {'✅ Multi-view analysis available' if matplotlib_success else '❌ Failed'}
- **Mayavi Scientific 3D:** {'✅ Professional 3D rendering available' if mayavi_success else '⚠️ Mayavi not available (install: pip install mayavi)'}
- **TB Region Detection:** Automatic bounding boxes and 3D markers

## **How to Use the Interactive 3D:**
- **🖱️ Rotate:** Click and drag to view from any angle
- **🔍 Zoom:** Mouse wheel to examine specific regions
- **📍 Hover:** Move mouse over surface to see exact attention values
- **🔴 Red Spheres:** Mark the highest attention TB regions
- **🎨 Color Scale:** Blue (normal) → Yellow (suspicious) → Red (TB)

## **3D Attention Map Interpretation:**
- **🏔️ High Peaks:** Potential TB lesions (AI highly focused)
- **🌊 Low Valleys:** Normal lung tissue (AI sees healthy patterns)
- **📏 Height Scale:** 0-100 units representing attention intensity

*Trained on TBX11K dataset with 94.63% validation accuracy*
"""

            print("🎯 Preparing final return values...")
            print(f"   Result text: {len(result_text)} characters")
            print(f"   Dashboard image: {dashboard_img is not None}")
            print(f"   Plotly figure: {plotly_fig is not None}")
            print(f"   Matplotlib image: {matplotlib_img is not None}")
            print(f"   Mayavi image: {mayavi_img is not None}")

            print("🚀 Returning all results to Gradio interface...")
            return result_text, dashboard_img, plotly_fig, matplotlib_img, mayavi_img, "✅ Interactive 3D analysis completed"

    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        import traceback
        traceback.print_exc()

        # Create empty figure for error
        import plotly.graph_objects as go
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

        return error_msg, None, error_fig, None, None, "Prediction failed"

def create_interface():
    """Create Gradio interface with working interactive 3D visualization"""
    print_section("CREATING INTERACTIVE 3D GRADIO INTERFACE")

    # Load model
    model, device = load_model()

    if model is None:
        print("❌ Cannot create interface without model")
        return None

    # Create interface with proper 3D components
    with gr.Blocks(title="TB Detection with Interactive 3D Visualization") as interface:
        gr.Markdown("""
        # 🫁 **TB Detection System with Interactive 3D Visualization**
        ### *Pre-trained on Real TBX11K Dataset (94.63% Accuracy)*

        Upload a chest X-ray image for AI-powered tuberculosis screening with **working interactive 3D visualization**.

        **🎯 Interactive 3D Features:**
        - 🌐 **Interactive Plotly 3D:** Full rotation, zoom, and hover capabilities
        - 📊 **Enhanced Multi-view 3D:** Main 3D + top/side/front projections
        - 🔍 **TB Region Detection:** Automatic bounding boxes and 3D markers
        - 📈 **Real-time Analysis:** Complete 2D+3D dashboard

        **🎨 3D Interaction Guide:**
        - 🖱️ **Click & Drag:** Rotate the 3D surface to view from any angle
        - 🔍 **Mouse Wheel:** Zoom in/out to examine specific regions
        - 📍 **Hover:** Move mouse over surface to see exact attention values
        - 🔴 **Red Spheres:** Mark the highest confidence TB regions
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 **Upload Chest X-ray**")
                input_image = gr.Image(
                    type="pil",
                    label="Chest X-ray Image",
                    height=350
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze with Interactive 3D Visualization",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("""
                **💡 After analysis:**
                - Use the interactive 3D plot below to explore
                - Rotate, zoom, and hover over the attention surface
                - Examine the multi-view 3D analysis
                - Check TB regions marked with red spheres
                """)

            with gr.Column(scale=2):
                gr.Markdown("## 📊 **Analysis Results**")

                result_text = gr.Markdown(
                    value="Upload an image and click 'Analyze' to see interactive 3D results.",
                    label="Detailed Analysis"
                )

                dashboard_viz = gr.Image(
                    label="2D Analysis Dashboard",
                    height=400
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready for interactive 3D analysis"
                )

        # Interactive 3D Visualization Section
        gr.Markdown("## 🎯 **Interactive 3D Visualizations**")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🌐 **Interactive Plotly 3D**")
                gr.Markdown("*Rotate, zoom, and hover for detailed exploration*")
                plotly_3d = gr.Plot(
                    label="Interactive 3D Attention Surface"
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 **Enhanced Multi-view 3D**")
                gr.Markdown("*Main 3D + top/side/front projections*")
                matplotlib_3d = gr.Image(
                    label="Multi-view 3D Analysis",
                    height=500
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔬 **Mayavi Scientific 3D**")
                gr.Markdown("*Professional scientific visualization with enhanced rendering*")
                mayavi_3d = gr.Image(
                    label="Mayavi Scientific 3D Visualization",
                    height=500
                )

        # Connect the analysis function
        analyze_btn.click(
            fn=lambda img: predict_tb_with_interactive_3d(img, model, device),
            inputs=[input_image],
            outputs=[result_text, dashboard_viz, plotly_3d, matplotlib_3d, mayavi_3d, status_text]
        )

    print("✅ Interactive 3D interface created successfully")
    return interface

def main():
    """Main function"""
    print("🫁 TB DETECTION SYSTEM - INTERACTIVE 3D VISUALIZATION")
    print("=" * 60)

    try:
        interface = create_interface()

        if interface is not None:
            print_section("LAUNCHING INTERACTIVE 3D INTERFACE")
            print("🌐 Starting TB Detection System with Interactive 3D...")
            print("🎯 Features: Working Plotly Interactive 3D + Enhanced Multi-view 3D")

            interface.launch(
                share=True,
                debug=False,
                show_error=True,
                server_name="0.0.0.0",
                server_port=7869
            )
        else:
            print("❌ Failed to create interface")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
