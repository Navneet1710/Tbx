#!/usr/bin/env python3
"""
TB Detection System - Auto-Running Version
Fixed and optimized version that runs automatically without manual intervention
"""

import os
import sys
import subprocess
import warnings
import time
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def install_packages():
    """Install required packages automatically"""
    print_section("INSTALLING REQUIRED PACKAGES")

    packages = [
        'torch', 'torchvision', 'timm', 'albumentations',
        'gradio', 'matplotlib', 'seaborn', 'opencv-python',
        'scikit-learn', 'pandas', 'numpy', 'pillow',
        'kagglehub', 'scipy'
    ]

    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
        except Exception as e:
            print(f"⚠️ Error with {package}: {e}")

    print("✅ Package installation completed!")

def verify_imports():
    """Verify all required imports work"""
    print_section("VERIFYING IMPORTS")

    try:
        # Core imports
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Core scientific packages imported")

        # Computer vision
        import cv2
        from PIL import Image
        print("✅ Computer vision packages imported")

        # Machine learning
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import torchvision.transforms as transforms
        import timm
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        print("✅ Machine learning packages imported")

        # Data augmentation
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print("✅ Data augmentation packages imported")

        # Web interface
        import gradio as gr
        print("✅ Gradio web interface imported")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Using device: {device}")

        return True, device

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None

def create_synthetic_data():
    """Create synthetic chest X-ray data for demo"""
    print_section("CREATING SYNTHETIC DATA")

    import numpy as np
    import cv2
    import tempfile

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='tb_auto_demo_')
    print(f"📁 Created temp directory: {temp_dir}")

    image_paths = []
    labels = []

    print("🎨 Generating synthetic chest X-ray images...")

    for i in range(40):  # Create 40 images
        # Create base chest X-ray like image
        img = np.random.randint(50, 200, (512, 512), dtype=np.uint8)

        # Add chest-like structures
        center_x, center_y = 256, 200

        # Add lung regions (darker areas)
        cv2.ellipse(img, (center_x - 80, center_y), (60, 100), 0, 0, 360, 30, -1)
        cv2.ellipse(img, (center_x + 80, center_y), (60, 100), 0, 0, 360, 30, -1)

        # Add ribs (brighter lines)
        for j in range(5):
            y_pos = center_y - 80 + j * 40
            cv2.line(img, (50, y_pos), (462, y_pos + 20), 150, 2)

        # Determine label and add features
        if i < 20:  # Normal images
            label = 0
        else:  # TB positive images - add bright spots
            label = 1
            # Add TB-like lesions
            for _ in range(np.random.randint(1, 4)):
                x = np.random.randint(180, 332)
                y = np.random.randint(150, 250)
                cv2.circle(img, (x, y), np.random.randint(8, 15), 220, -1)

        # Save image
        img_path = os.path.join(temp_dir, f'synthetic_xray_{i:03d}.png')
        cv2.imwrite(img_path, img)

        image_paths.append(img_path)
        labels.append(label)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/40 images")

    print(f"✅ Created {len(image_paths)} synthetic images")
    print(f"   Normal: {labels.count(0)}, TB: {labels.count(1)}")

    return image_paths, labels, temp_dir

def create_model_and_transforms(device):
    """Create the TB detection model and data transforms"""
    print_section("CREATING MODEL AND TRANSFORMS")

    import torch
    import torch.nn as nn
    import timm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Create data transforms
    print("🔄 Creating data transforms...")
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create model
    print("🧠 Creating TB detection model...")

    class TBDetectionModel(nn.Module):
        def __init__(self, num_classes=2, backbone='efficientnet_b0'):
            super(TBDetectionModel, self).__init__()

            # Load pretrained backbone
            self.backbone = timm.create_model(
                backbone, pretrained=True, features_only=True, out_indices=[4]
            )

            # Get feature dimensions
            dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
            with torch.no_grad():
                features = self.backbone(dummy_input)
                feature_dim = features[0].shape[1]

            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool2d(1)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )

            # Attention for visualization
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # Extract features
            features = self.backbone(x)[0]  # Get last feature map

            # Apply attention
            attention_map = self.attention(features)
            attended_features = features * attention_map

            # Global pooling and classification
            pooled = self.gap(attended_features).flatten(1)
            output = self.classifier(pooled)

            return {
                'logits': output,
                'attention_map': attention_map,
                'features': features
            }

    model = TBDetectionModel(num_classes=2, backbone='efficientnet_b0')
    model = model.to(device)

    # Test model
    dummy_input = torch.randn(2, 3, 224, 224, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"✅ Model created successfully")
        print(f"   Output shape: {test_output['logits'].shape}")
        print(f"   Attention shape: {test_output['attention_map'].shape}")

    return model, train_transform, val_transform

def create_dataset_and_loaders(image_paths, labels, train_transform, val_transform):
    """Create dataset and data loaders"""
    print_section("CREATING DATASETS AND LOADERS")

    import torch
    from torch.utils.data import Dataset, DataLoader
    import cv2
    import numpy as np
    from sklearn.model_selection import train_test_split

    class TBDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            # Load image
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = self.labels[idx]

            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'path': img_path
            }

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"📊 Data split:")
    print(f"   Training samples: {len(train_paths)}")
    print(f"   Validation samples: {len(val_paths)}")

    # Create datasets
    train_dataset = TBDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TBDataset(val_paths, val_labels, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"✅ Created data loaders")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    """Train the TB detection model with early stopping"""
    print_section("TRAINING MODEL")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 3

    print(f"🚀 Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs['logits'], labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"  ⚠️ Error in batch {batch_idx}: {e}")
                continue

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(images)
                    loss = criterion(outputs['logits'], labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs['logits'], 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                except Exception as e:
                    print(f"  ⚠️ Error in validation: {e}")
                    continue

        # Calculate metrics
        if len(train_loader) > 0 and len(val_loader) > 0:
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / max(train_total, 1)
            val_acc = 100. * val_correct / max(val_total, 1)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"✅ New best model (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break

            scheduler.step(val_loss)

    print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model, history

def save_model(model, history, filepath='tb_detection_model.pth'):
    """Save the trained model"""
    print_section("SAVING MODEL")

    import torch

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'model_config': {
            'num_classes': 2,
            'backbone': 'efficientnet_b0'
        }
    }

    torch.save(checkpoint, filepath)
    print(f"✅ Model saved to {filepath}")
    return filepath

def create_gradio_interface(model_path, device):
    """Create and launch Gradio web interface"""
    print_section("CREATING WEB INTERFACE")

    import gradio as gr
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import io

    # Load model for inference
    def load_model_for_inference():
        try:
            checkpoint = torch.load(model_path, map_location=device)

            # Recreate model (we need to import the class again)
            import timm
            import torch.nn as nn

            class TBDetectionModel(nn.Module):
                def __init__(self, num_classes=2, backbone='efficientnet_b0'):
                    super(TBDetectionModel, self).__init__()
                    self.backbone = timm.create_model(
                        backbone, pretrained=True, features_only=True, out_indices=[4]
                    )
                    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
                    with torch.no_grad():
                        features = self.backbone(dummy_input)
                        feature_dim = features[0].shape[1]
                    self.gap = nn.AdaptiveAvgPool2d(1)
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(feature_dim, 256),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(256),
                        nn.Dropout(0.2),
                        nn.Linear(256, num_classes)
                    )
                    self.attention = nn.Sequential(
                        nn.Conv2d(feature_dim, 64, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 1, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    features = self.backbone(x)[0]
                    attention_map = self.attention(features)
                    attended_features = features * attention_map
                    pooled = self.gap(attended_features).flatten(1)
                    output = self.classifier(pooled)
                    return {
                        'logits': output,
                        'attention_map': attention_map,
                        'features': features
                    }

            model = TBDetectionModel()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return None

    inference_model = load_model_for_inference()

    def predict_tb(image):
        """Predict TB from chest X-ray image"""
        if inference_model is None:
            return "❌ Model not loaded", None, "Model loading failed"

        if image is None:
            return "❌ No image provided", None, "Please upload an image"

        try:
            # Preprocess image
            if isinstance(image, Image.Image):
                image = np.array(image)

            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize and normalize
            image_resized = cv2.resize(image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0

            # Apply normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_normalized = (image_normalized - mean) / std

            # Convert to tensor with explicit float32 type
            image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = inference_model(image_tensor)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][int(predicted_class)].item()

                # Get attention map
                attention = outputs['attention_map'][0, 0].cpu().numpy()
                attention = cv2.resize(attention, (224, 224))

                # Create attention visualization
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(image_resized)
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(image_resized)
                plt.imshow(attention, alpha=0.5, cmap='jet')
                plt.title('Attention Heatmap')
                plt.axis('off')

                plt.tight_layout()

                # Save plot to bytes
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()

                # Convert to PIL Image
                attention_img = Image.open(buf)

                # Create result text
                class_names = ['Normal', 'TB Positive']
                result_class = class_names[int(predicted_class)]

                result_text = f"""
🔍 **Prediction Result:**
- **Classification:** {result_class}
- **Confidence:** {confidence:.2%}
- **Risk Level:** {'High' if predicted_class == 1 and confidence > 0.7 else 'Moderate' if predicted_class == 1 else 'Low'}

⚠️ **Important:** This is an AI screening tool for educational purposes only.
Always consult healthcare professionals for proper medical diagnosis.
"""

                return result_text, attention_img, f"Analysis completed successfully"

        except Exception as e:
            return f"❌ Error during prediction: {str(e)}", None, "Prediction failed"

    # Create Gradio interface
    print("🌐 Creating Gradio interface...")

    with gr.Blocks(title="TB Detection System") as interface:
        gr.Markdown("""
        # 🫁 TB Detection System

        Upload a chest X-ray image for AI-powered tuberculosis screening.

        **Features:**
        - Deep learning-based TB detection
        - Attention mechanism visualization
        - Real-time analysis
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload Chest X-ray Image",
                    height=300
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze X-ray",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                result_text = gr.Markdown(label="Analysis Result")

                with gr.Row():
                    attention_viz = gr.Image(
                        label="Attention Visualization",
                        height=300
                    )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )

        # Connect the prediction function
        analyze_btn.click(
            fn=predict_tb,
            inputs=[input_image],
            outputs=[result_text, attention_viz, status_text]
        )

        gr.Markdown("""
        ---
        **Disclaimer:** This tool is for educational and screening purposes only.
        Always consult qualified healthcare professionals for medical diagnosis.
        """)

    print("✅ Gradio interface created successfully")
    return interface

def main():
    """Main function that runs the entire TB detection pipeline automatically"""
    print("🫁 TB DETECTION SYSTEM - AUTO RUNNER")
    print("=" * 60)
    print("This script will automatically:")
    print("1. Install required packages")
    print("2. Create synthetic training data")
    print("3. Train a TB detection model")
    print("4. Launch a web interface")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Install packages
        install_packages()

        # Step 2: Verify imports
        imports_ok, device = verify_imports()
        if not imports_ok:
            print("❌ Failed to import required packages. Please check installation.")
            return

        # Step 3: Create synthetic data
        image_paths, labels, temp_dir = create_synthetic_data()

        # Step 4: Create model and transforms
        model, train_transform, val_transform = create_model_and_transforms(device)

        # Step 5: Create datasets and loaders
        train_loader, val_loader = create_dataset_and_loaders(
            image_paths, labels, train_transform, val_transform
        )

        # Step 6: Train model
        trained_model, history = train_model(model, train_loader, val_loader, device, num_epochs=5)

        # Step 7: Save model
        model_path = save_model(trained_model, history, 'tb_detection_auto_model.pth')

        # Step 8: Create and launch interface
        interface = create_gradio_interface(model_path, device)

        # Calculate total time
        total_time = time.time() - start_time

        print_section("SYSTEM READY")
        print(f"✅ TB Detection System is ready!")
        print(f"⏱️ Total setup time: {total_time:.1f} seconds")
        print(f"📁 Temporary data directory: {temp_dir}")
        print(f"💾 Model saved to: {model_path}")
        print(f"🌐 Launching web interface...")

        # Launch interface
        interface.launch(
            share=True,  # Create public link
            debug=False,  # Disable debug mode for cleaner output
            show_error=True,  # Show errors in interface
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860  # Default Gradio port
        )

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

        # Try to launch a basic interface anyway
        try:
            print("\n🔄 Attempting to launch basic interface...")
            import gradio as gr

            def basic_predict(image):
                if image is None:
                    return "⚠️ No image uploaded. Model not trained yet.", None, "Demo mode"
                return "⚠️ Model not trained yet. Please run the full pipeline.", None, "Demo mode"

            with gr.Blocks(title="TB Detection System - Demo") as demo_interface:
                gr.Markdown("# 🫁 TB Detection System - Demo Mode")
                gr.Markdown("⚠️ Training failed, but you can still see the interface layout.")

                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Chest X-ray")
                    analyze_btn = gr.Button("Analyze (Demo)")

                result_text = gr.Markdown()

                analyze_btn.click(
                    fn=basic_predict,
                    inputs=[input_image],
                    outputs=[result_text]
                )

            demo_interface.launch(share=True, debug=False)

        except Exception as demo_error:
            print(f"❌ Failed to launch demo interface: {demo_error}")

if __name__ == "__main__":
    # Auto-run the entire pipeline
    main()
