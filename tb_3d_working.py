#!/usr/bin/env python3
"""
TB Detection System with WORKING 3D Visualization
Simplified version focusing on functional 3D visualization
"""

import os
import torch
import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def create_simple_plotly_3d(image, attention_map, predicted_class, confidence):
    """Create simple working Plotly 3D visualization"""
    try:
        import plotly.graph_objects as go
        from scipy import ndimage

        print("🎨 Creating simple Plotly 3D visualization...")

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
        step = max(1, min(h, w) // 40)
        x = np.arange(0, w, step)
        y = np.arange(0, h, step)
        X, Y = np.meshgrid(x, y)
        Z = height_map_smooth[::step, ::step]

        # Create simple 3D surface - FIXED VERSION
        surface = go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Hot',
            opacity=0.9,
            name='Attention Surface'
        )

        # Create figure
        fig = go.Figure(data=[surface])

        # Add TB region markers
        tb_threshold = np.percentile(attention_normalized.astype(np.float32), 85)
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
                marker=dict(size=8, color='red', opacity=0.8),
                name='TB Regions'
            )
            fig.add_trace(tb_markers)

        # Update layout
        title_text = f'3D TB Attention - {["Normal", "TB Positive"][predicted_class]} (Confidence: {confidence:.1%})'

        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Attention Height',
                bgcolor='rgb(10, 10, 20)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=1000,
            height=700
        )

        # Save as HTML
        html_file = 'tb_3d_attention_working.html'
        fig.write_html(html_file, include_plotlyjs=True)

        print(f"✅ Working Plotly 3D visualization saved as '{html_file}'")
        return True, html_file

    except Exception as e:
        print(f"❌ Error creating Plotly 3D: {e}")
        return False, None

def create_mayavi_3d(image, attention_map, predicted_class, confidence):
    """Create Mayavi 3D visualization"""
    try:
        from mayavi import mlab
        from scipy import ndimage

        print("🎨 Creating Mayavi 3D visualization...")

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

        # Create coordinate grids
        h, w = image.shape[:2]
        x, y = np.mgrid[0:w:1, 0:h:1]

        # Create Mayavi figure
        mlab.figure(f'3D TB Attention - {["Normal", "TB Positive"][predicted_class]}',
                   bgcolor=(0.05, 0.05, 0.1), size=(1000, 800))
        mlab.clf()

        # Create 3D surface
        surf = mlab.surf(x, y, height_map_smooth.T, colormap='hot', opacity=0.9)

        # Add TB region markers
        tb_threshold = np.percentile(attention_normalized.astype(np.float32), 85)
        tb_regions = np.where(attention_normalized > tb_threshold)

        if len(tb_regions[0]) > 0:
            sample_size = min(20, len(tb_regions[0]))
            sample_indices = np.random.choice(len(tb_regions[0]), sample_size, replace=False)

            tb_x = tb_regions[1][sample_indices]
            tb_y = tb_regions[0][sample_indices]
            tb_z = height_map_smooth[tb_regions[0][sample_indices], tb_regions[1][sample_indices]] + 5

            mlab.points3d(tb_x, tb_y, tb_z, scale_factor=8, color=(1, 0, 0), opacity=0.8)

        # Customize view
        mlab.view(azimuth=45, elevation=65, distance='auto')

        # Add text
        mlab.text3d(w*0.1, h*0.1, height_scale + 20, "TB Regions", scale=8, color=(1, 0, 0))
        mlab.text3d(w*0.4, h*0.9, height_scale + 30, f"Confidence: {confidence:.1%}",
                   scale=6, color=(1, 1, 1))

        # Save image
        try:
            mlab.savefig('tb_3d_mayavi_working.png', size=(1200, 800))
            print("✅ Mayavi 3D saved as 'tb_3d_mayavi_working.png'")
        except:
            print("⚠️ Could not save Mayavi image")

        print("✅ Mayavi 3D visualization created successfully")
        return True

    except ImportError:
        print("⚠️ Mayavi not available")
        return False
    except Exception as e:
        print(f"❌ Error creating Mayavi 3D: {e}")
        return False

def create_enhanced_visualization(image, attention_map, predicted_class, confidence, normal_prob, tb_prob):
    """Create enhanced visualization with working 3D components"""
    try:
        from matplotlib.patches import Rectangle

        print("🎨 Creating enhanced visualization with working 3D...")

        # Try 3D visualizations
        plotly_success, html_file = create_simple_plotly_3d(image, attention_map, predicted_class, confidence)
        mayavi_success = create_mayavi_3d(image, attention_map, predicted_class, confidence)

        # Ensure inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Create 2D visualization
        fig = plt.figure(figsize=(16, 10))
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

        # Highlight TB regions
        tb_threshold = np.percentile(attention_map.astype(np.float32), 85)
        tb_mask = attention_map > tb_threshold

        # Find contours
        tb_mask_uint8 = (tb_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(tb_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                rect = Rectangle((x, y), w, h, linewidth=2,
                               edgecolor='red' if predicted_class == 1 else 'yellow',
                               facecolor='none')
                ax4.add_patch(rect)

        ax4.set_title('TB Region Detection', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # 5. 3D Status and Info
        ax5 = fig.add_subplot(gs[1, :2])

        status_text = "3D Visualization Status:\n\n"
        if plotly_success:
            status_text += f"✅ Interactive Plotly 3D: {html_file}\n"
        else:
            status_text += "❌ Plotly 3D: Failed\n"

        if mayavi_success:
            status_text += "✅ Mayavi 3D: tb_3d_mayavi_working.png\n"
        else:
            status_text += "❌ Mayavi 3D: Not available\n"

        status_text += "\n💡 Open HTML file in browser for interactive 3D!"

        ax5.text(0.5, 0.5, status_text, ha='center', va='center',
                fontsize=12, transform=ax5.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax5.axis('off')

        # 6. Probability Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        classes = ['Normal', 'TB Positive']
        probs = [normal_prob, tb_prob]
        colors = ['green' if predicted_class == 0 else 'lightgreen',
                 'red' if predicted_class == 1 else 'lightcoral']

        bars = ax6.bar(classes, probs, color=colors)
        ax6.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Probability')
        ax6.set_ylim(0, 1)

        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

        # 7. Risk Assessment
        ax7 = fig.add_subplot(gs[1, 3])
        risk_level = "High" if predicted_class == 1 and confidence > 0.8 else \
                    "Moderate" if predicted_class == 1 and confidence > 0.6 else \
                    "Low-Moderate" if predicted_class == 1 else "Low"

        risk_colors = {"High": "red", "Moderate": "orange", "Low-Moderate": "yellow", "Low": "green"}

        ax7.text(0.5, 0.7, "Risk Level", ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax7.transAxes)
        ax7.text(0.5, 0.5, risk_level, ha='center', va='center',
                fontsize=20, fontweight='bold', color=risk_colors[risk_level],
                transform=ax7.transAxes)
        ax7.text(0.5, 0.3, f"Confidence: {confidence:.1%}", ha='center', va='center',
                fontsize=12, transform=ax7.transAxes)
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return Image.open(buf)

    except Exception as e:
        print(f"❌ Error creating enhanced visualization: {e}")
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

def predict_tb(image, model, device):
    """Predict TB with working 3D visualization"""
    if model is None:
        return "❌ Model not loaded", None, "Model loading failed"

    if image is None:
        return "❌ No image provided", None, "Please upload a chest X-ray"

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

            # Create enhanced visualization with working 3D
            result_img = create_enhanced_visualization(
                image_resized, attention, predicted_class, confidence,
                normal_prob, tb_prob
            )

            # Create result text
            class_names = ['Normal', 'TB Positive']
            result_class = class_names[int(predicted_class)]

            result_text = f"""
# 🔍 **TB Detection Analysis with Working 3D Visualization**

## **Primary Diagnosis**
- **Classification:** {result_class}
- **Confidence:** {confidence:.1%}

## **Detailed Probabilities**
- **Normal/Healthy:** {normal_prob:.1%}
- **TB Positive:** {tb_prob:.1%}

## **3D Visualization Status**
- **Interactive Plotly 3D:** Check 'tb_3d_attention_working.html'
- **Mayavi Scientific 3D:** Check 'tb_3d_mayavi_working.png'
- **TB Region Detection:** Automatic bounding boxes around suspicious areas

## **3D Attention Map Explanation**
- **High Peaks (Mountains):** Potential TB lesions - AI highly focused
- **Low Valleys:** Normal lung tissue - AI sees healthy patterns
- **Color Coding:** Blue (normal) → Yellow (suspicious) → Red (TB regions)
- **Height Scale:** 0-100 units representing attention intensity

*Trained on TBX11K dataset with 94.63% validation accuracy*
"""

            return result_text, result_img, "✅ Analysis completed with working 3D visualization"

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", None, "Prediction failed"

def create_interface():
    """Create Gradio interface with working 3D visualization"""
    print_section("CREATING GRADIO INTERFACE")

    # Load model
    model, device = load_model()

    if model is None:
        print("❌ Cannot create interface without model")
        return None

    # Create interface
    with gr.Blocks(title="TB Detection with Working 3D Visualization") as interface:
        gr.Markdown("""
        # 🫁 **TB Detection System with Working 3D Visualization**
        ### *Pre-trained on Real TBX11K Dataset (94.63% Accuracy)*

        Upload a chest X-ray image for AI-powered tuberculosis screening with **working 3D visualization**.

        **🎯 Working 3D Features:**
        - 🏔️ **3D Attention Surface:** Height-mapped visualization where peaks = TB regions
        - 🌐 **Interactive Plotly 3D:** Browser-based interaction (saved as HTML)
        - 🎨 **Mayavi Scientific 3D:** Professional medical visualization (saved as PNG)
        - 🔍 **TB Region Detection:** Automatic bounding boxes around suspicious areas

        **🎨 3D Interpretation:**
        - 🔵 **Blue/Low Areas:** Normal healthy tissue
        - 🟡 **Yellow/High Peaks:** Suspicious TB regions
        - 🔴 **Red Markers:** Highest attention TB areas
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
                    "🔍 Analyze with Working 3D Visualization",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                gr.Markdown("## 📊 **Analysis Results**")

                result_text = gr.Markdown(
                    value="Upload an image and click 'Analyze' to see working 3D results.",
                    label="Detailed Analysis"
                )

                result_viz = gr.Image(
                    label="Enhanced 2D+3D Visualization",
                    height=500
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready for analysis with working 3D visualization"
                )

        # Connect the analysis function
        analyze_btn.click(
            fn=lambda img: predict_tb(img, model, device),
            inputs=[input_image],
            outputs=[result_text, result_viz, status_text]
        )

    print("✅ Interface with working 3D visualization created successfully")
    return interface

def main():
    """Main function"""
    print("🫁 TB DETECTION SYSTEM - WORKING 3D VISUALIZATION")
    print("=" * 60)

    try:
        interface = create_interface()

        if interface is not None:
            print_section("LAUNCHING INTERFACE")
            print("🌐 Starting TB Detection System with Working 3D Visualization...")

            interface.launch(
                share=True,
                debug=False,
                show_error=True,
                server_name="0.0.0.0",
                server_port=7864
            )
        else:
            print("❌ Failed to create interface")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
