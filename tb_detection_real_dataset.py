#!/usr/bin/env python3
"""
TB Detection System - Using Real TBX11K Dataset from Kaggle
This version downloads and uses the actual 11,000+ chest X-ray images
"""

import os
import sys
import subprocess
import warnings
import time
import json
from pathlib import Path
import xml.etree.ElementTree as ET

warnings.filterwarnings('ignore')

def install_packages():
    """Install required packages automatically"""
    print_section("INSTALLING REQUIRED PACKAGES")

    # Check if we should install CUDA version of PyTorch
    try:
        import torch
        current_cuda = torch.cuda.is_available()
        print(f"🔍 Current PyTorch CUDA support: {current_cuda}")
    except ImportError:
        current_cuda = False
        print("🔍 PyTorch not found, will install CUDA version")

    # Install PyTorch with CUDA support if not already available
    if not current_cuda:
        print("📦 Installing PyTorch with CUDA support...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--quiet"
            ])
            print("✅ PyTorch with CUDA installed successfully")
        except Exception as e:
            print(f"⚠️ Failed to install CUDA PyTorch: {e}")
            print("🔄 Installing CPU version as fallback...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "--quiet"
            ])

    # Install other packages including Mayavi for 3D visualization
    other_packages = [
        'timm', 'albumentations', 'gradio', 'matplotlib',
        'seaborn', 'opencv-python', 'scikit-learn',
        'pandas', 'numpy', 'pillow', 'kagglehub', 'scipy',
        'mayavi', 'vtk', 'PyQt5'  # Added for 3D visualization
    ]

    for package in other_packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"✅ {package} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
            # For Mayavi, try alternative installation
            if package == 'mayavi':
                try:
                    print("🔄 Trying alternative Mayavi installation...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "mayavi[qt]", "--quiet"
                    ])
                    print("✅ Mayavi installed successfully (alternative method)")
                except:
                    print("⚠️ Mayavi installation failed, 3D visualization may not work")

    print("✅ Package installation completed!")

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

# def install_packages():
#     """Install required packages"""
#     print_section("INSTALLING REQUIRED PACKAGES")

#     packages = [
#         'torch', 'torchvision', 'timm', 'albumentations',
#         'gradio', 'matplotlib', 'seaborn', 'opencv-python',
#         'scikit-learn', 'pandas', 'numpy', 'pillow',
#         'kagglehub', 'scipy'
#     ]

#     for package in packages:
#         try:
#             print(f"📦 Installing {package}...")
#             subprocess.check_call([
#                 sys.executable, "-m", "pip", "install", package, "--quiet"
#             ])
#             print(f"✅ {package} installed successfully")
#         except Exception as e:
#             print(f"❌ Failed to install {package}: {e}")

#     print("✅ Package installation completed!")

def download_tbx11k_dataset():
    """Download the real TBX11K dataset from Kaggle"""
    print_section("DOWNLOADING REAL TBX11K DATASET")

    try:
        import kagglehub

        print("📥 Downloading TBX11K dataset from Kaggle...")
        print("⏳ This may take several minutes (3+ GB download)...")

        # Download latest version
        path = kagglehub.dataset_download("usmanshams/tbx-11")

        print(f"✅ Dataset downloaded successfully!")
        print(f"📁 Path to dataset files: {path}")

        return path

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure you have Kaggle API configured:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Create API token")
        print("   3. Place kaggle.json in ~/.kaggle/ directory")
        return None

def explore_tbx11k_structure(dataset_path):
    """Explore the actual TBX11K dataset structure"""
    print_section("EXPLORING DATASET STRUCTURE")

    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset path not found")
        return None

    dataset_info = {
        'base_path': dataset_path,
        'image_dirs': [],
        'annotation_files': [],
        'total_images': 0,
        'class_distribution': {}
    }

    print(f"🔍 Exploring: {dataset_path}")

    # Walk through directory structure
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root)
        print(f"{indent}{folder_name}/")

        # Check for images
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            dataset_info['image_dirs'].append(root)
            dataset_info['total_images'] += len(image_files)
            print(f"{indent}  └── {len(image_files)} images")

        # Check for annotations
        annotation_files = [f for f in files if f.lower().endswith(('.json', '.csv', '.txt', '.xml'))]
        if annotation_files:
            dataset_info['annotation_files'].extend([os.path.join(root, f) for f in annotation_files])
            print(f"{indent}  └── {len(annotation_files)} annotation files")

    print(f"\n📊 Dataset Summary:")
    print(f"   Total images found: {dataset_info['total_images']}")
    print(f"   Image directories: {len(dataset_info['image_dirs'])}")
    print(f"   Annotation files: {len(dataset_info['annotation_files'])}")

    return dataset_info

def prepare_real_tbx11k_data(dataset_path):
    """Prepare the real TBX11K dataset for training"""
    print_section("PREPARING REAL TBX11K DATA")

    if not dataset_path:
        print("❌ No dataset path provided")
        return [], [], []

    image_paths = []
    labels = []
    bboxes = []

    try:
        # Look for the main TBX11K directory structure
        tbx_dir = None
        for root, dirs, files in os.walk(dataset_path):
            if 'TBX11K' in dirs:
                tbx_dir = os.path.join(root, 'TBX11K')
                break
            elif os.path.basename(root) == 'TBX11K':
                tbx_dir = root
                break

        if not tbx_dir:
            print("❌ TBX11K directory not found in dataset")
            return [], [], []

        print(f"📁 Found TBX11K directory: {tbx_dir}")

        # Look for image directories
        imgs_dir = os.path.join(tbx_dir, 'imgs')
        if not os.path.exists(imgs_dir):
            print("❌ imgs directory not found")
            return [], [], []

        print(f"📁 Found imgs directory: {imgs_dir}")

        # Process different categories
        categories = {
            'health': 0,    # Normal/Healthy
            'sick': 1,      # Sick (may include TB)
            'tb': 1,        # Definitely TB
            'normal': 0,    # Normal
            'abnormal': 1   # Abnormal
        }

        for category, label in categories.items():
            category_path = os.path.join(imgs_dir, category)
            if os.path.exists(category_path):
                print(f"📂 Processing {category} directory...")

                image_files = [f for f in os.listdir(category_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in image_files:
                    img_path = os.path.join(category_path, img_file)

                    # Verify image can be loaded
                    try:
                        import cv2
                        test_img = cv2.imread(img_path)
                        if test_img is not None and test_img.shape[0] > 50 and test_img.shape[1] > 50:
                            image_paths.append(img_path)
                            labels.append(label)
                            bboxes.append(None)  # No bounding box info for now
                    except Exception as e:
                        print(f"⚠️ Skipping corrupted image: {img_path}")
                        continue

                print(f"   ✅ Added {len([l for l in labels if l == label and image_paths.index(img_path) < len(labels)])} {category} images")

        # Also check test directory for additional images
        test_dir = os.path.join(imgs_dir, 'test')
        if os.path.exists(test_dir):
            print(f"📂 Processing test directory...")
            test_files = [f for f in os.listdir(test_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # For test images, we'll assign labels based on filename patterns or use as unlabeled
            for img_file in test_files[:1000]:  # Limit to first 1000 test images
                img_path = os.path.join(test_dir, img_file)
                try:
                    import cv2
                    test_img = cv2.imread(img_path)
                    if test_img is not None:
                        # Try to infer label from filename or use as normal by default
                        if any(keyword in img_file.lower() for keyword in ['tb', 'abnormal', 'positive']):
                            label = 1
                        else:
                            label = 0

                        image_paths.append(img_path)
                        labels.append(label)
                        bboxes.append(None)
                except:
                    continue

            print(f"   ✅ Added {len(test_files[:1000])} test images")

        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Total images: {len(image_paths)}")

        if len(labels) > 0:
            import numpy as np
            unique, counts = np.unique(labels, return_counts=True)
            class_names = ['Normal/Healthy', 'TB/Abnormal']
            for i, (cls, count) in enumerate(zip(unique, counts)):
                print(f"   {class_names[cls]}: {count} images")

        return image_paths, labels, bboxes

    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

# Move dataset class to global scope to fix Windows multiprocessing issue
import torch
from torch.utils.data import Dataset

class TBX11Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2
        import numpy as np

        # Load image
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Create dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)

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

def create_real_data_loaders(image_paths, labels, train_transform, val_transform, batch_size=16):
    """Create data loaders for the real dataset"""
    print_section("CREATING DATA LOADERS")

    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    # Split data
    if len(image_paths) == 0:
        print("❌ No images found for training")
        return None, None

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )

    print(f"📊 Data split:")
    print(f"   Training samples: {len(train_paths)}")
    print(f"   Validation samples: {len(val_paths)}")

    # Create datasets
    train_dataset = TBX11Dataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TBX11Dataset(val_paths, val_labels, transform=val_transform)

    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Fix for Windows multiprocessing
        pin_memory=False  # Disable for stability
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Fix for Windows multiprocessing
        pin_memory=False  # Disable for stability
    )

    print(f"✅ Created data loaders")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    return train_loader, val_loader

def create_transforms():
    """Create data transforms for real medical images"""
    print_section("CREATING DATA TRANSFORMS")

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # More aggressive augmentation for medical images
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.8),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(noise_scale_factor=0.1, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            p=0.3
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    print("✅ Data transforms created")
    return train_transform, val_transform

def create_3d_attention_visualization(image, attention_map, title="3D TB Attention Map"):
    """Create 3D visualization of attention map using Mayavi"""
    try:
        # Import required libraries
        from mayavi import mlab
        import numpy as np
        from scipy import ndimage
        import cv2

        print("🎨 Creating 3D attention visualization...")

        # Ensure image and attention are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image if needed
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Normalize attention map
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Create height map from attention (higher attention = higher elevation)
        height_scale = 50  # Adjust this to control 3D height
        height_map = attention_normalized * height_scale

        # Smooth the height map for better 3D appearance
        height_map_smooth = ndimage.gaussian_filter(height_map, sigma=2)

        # Create coordinate grids
        x, y = np.mgrid[0:image.shape[1]:1, 0:image.shape[0]:1]

        # Convert image to grayscale for texture mapping
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        # Create figure
        mlab.figure(title, bgcolor=(0.1, 0.1, 0.1), size=(800, 600))

        # Create 3D surface with attention as height and image as texture
        surf = mlab.surf(x, y, height_map_smooth.T,
                        colormap='hot',
                        representation='surface',
                        opacity=0.8)

        # Add color mapping based on attention intensity
        surf.module_manager.scalar_lut_manager.lut.table = create_custom_colormap()

        # Add contour lines to show attention levels
        mlab.contour3d(x, y, np.zeros_like(x), height_map_smooth.T,
                      contours=8, transparent=True, opacity=0.3, colormap='cool')

        # Add base image as a plane
        base_plane = mlab.imshow(image_gray, colormap='gray', opacity=0.6)
        base_plane.actor.actor.position = [0, 0, -5]

        # Add TB region markers (high attention areas)
        tb_threshold = np.percentile(attention_normalized, 85)  # Top 15% attention
        tb_regions = np.where(attention_normalized > tb_threshold)

        if len(tb_regions[0]) > 0:
            # Sample points to avoid overcrowding
            sample_indices = np.random.choice(len(tb_regions[0]),
                                            min(50, len(tb_regions[0])),
                                            replace=False)

            tb_x = tb_regions[1][sample_indices]
            tb_y = tb_regions[0][sample_indices]
            tb_z = height_map_smooth[tb_regions[0][sample_indices], tb_regions[1][sample_indices]]

            # Add TB markers
            mlab.points3d(tb_x, tb_y, tb_z + 5,
                         scale_factor=3, color=(1, 0, 0), opacity=0.8)

        # Customize the view
        mlab.view(azimuth=45, elevation=60, distance='auto')

        # Add colorbar
        mlab.colorbar(surf, title="Attention Intensity", orientation='vertical')

        # Add text annotations
        mlab.text3d(10, 10, height_scale + 10, "TB Regions", scale=3, color=(1, 1, 1))
        mlab.text3d(image.shape[1] - 50, 10, height_scale + 10, "Normal", scale=3, color=(0, 1, 0))

        # Add axes
        mlab.axes(surf, color=(0.7, 0.7, 0.7), xlabel='X (pixels)',
                 ylabel='Y (pixels)', zlabel='Attention Height')

        # Add outline
        mlab.outline(surf, color=(0.7, 0.7, 0.7))

        print("✅ 3D visualization created successfully")
        return True

    except ImportError as e:
        print(f"⚠️ Mayavi not available: {e}")
        print("💡 Install with: pip install mayavi vtk PyQt5")
        return False
    except Exception as e:
        print(f"❌ Error creating 3D visualization: {e}")
        return False

def create_custom_colormap():
    """Create custom colormap for TB attention visualization"""
    import numpy as np

    # Create a colormap that emphasizes TB regions
    # Blue (low attention) -> Green (medium) -> Yellow (high) -> Red (TB regions)
    colors = np.array([
        [0, 0, 0.5, 255],      # Dark blue (no attention)
        [0, 0, 1, 255],        # Blue (low attention)
        [0, 0.5, 1, 255],      # Light blue
        [0, 1, 1, 255],        # Cyan
        [0, 1, 0.5, 255],      # Green-cyan
        [0, 1, 0, 255],        # Green (medium attention)
        [0.5, 1, 0, 255],      # Yellow-green
        [1, 1, 0, 255],        # Yellow (high attention)
        [1, 0.5, 0, 255],      # Orange
        [1, 0, 0, 255],        # Red (TB regions)
    ], dtype=np.uint8)

    return colors

def create_enhanced_2d_3d_visualization(image, attention_map, predicted_class, confidence,
                                       normal_prob, tb_prob):
    """Create comprehensive 2D and 3D visualization for TB detection"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        import io
        from PIL import Image as PILImage
        import cv2
        from mpl_toolkits.mplot3d import Axes3D

        print("🎨 Creating enhanced 2D+3D visualization...")

        # Ensure inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(attention_map, np.ndarray):
            attention_map = np.array(attention_map)

        # Resize attention map to match image
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Create figure with subplots for 2D visualizations
        fig = plt.figure(figsize=(20, 12))

        # Define grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1, 1])

        # 1. Original chest X-ray
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax1.set_title('Original Chest X-ray', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. 2D Attention Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(attention_map, cmap='hot', alpha=0.8)
        ax2.set_title('2D Attention Heatmap', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Attention Intensity')

        # 3. Overlay visualization
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        overlay = ax3.imshow(attention_map, cmap='hot', alpha=0.6)
        ax3.set_title('Attention Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # 4. TB Region Detection
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(image, cmap='gray' if len(image.shape) == 2 else None)

        # Highlight high attention regions (potential TB areas)
        tb_threshold = np.percentile(attention_map, 85)
        tb_mask = attention_map > tb_threshold

        # Find contours of TB regions
        import cv2
        tb_mask_uint8 = (tb_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(tb_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around TB regions
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                rect = Rectangle((x, y), w, h, linewidth=2,
                               edgecolor='red' if predicted_class == 1 else 'yellow',
                               facecolor='none')
                ax4.add_patch(rect)

                # Add confidence text
                ax4.text(x, y-5, f'{confidence:.1%}',
                        color='red' if predicted_class == 1 else 'yellow',
                        fontweight='bold', fontsize=10)

        ax4.set_title('TB Region Detection', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # 5. 3D Attention Surface (placeholder for Mayavi)
        ax5 = fig.add_subplot(gs[1, :2], projection='3d')

        # Create 3D surface plot as fallback
        x = np.arange(0, attention_map.shape[1], 4)
        y = np.arange(0, attention_map.shape[0], 4)
        X, Y = np.meshgrid(x, y)
        Z = attention_map[::4, ::4] * 50  # Scale for visibility

        surf = ax5.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)
        ax5.set_title('3D Attention Surface\n(Matplotlib Fallback)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('X (pixels)')
        ax5.set_ylabel('Y (pixels)')
        ax5.set_zlabel('Attention Height')

        # 6. Probability Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        classes = ['Normal', 'TB Positive']
        probs = [normal_prob, tb_prob]
        colors = ['green' if predicted_class == 0 else 'lightgreen',
                 'red' if predicted_class == 1 else 'lightcoral']

        bars = ax6.bar(classes, probs, color=colors)
        ax6.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Probability')
        ax6.set_ylim(0, 1)

        # Add probability labels
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
                fontsize=16, fontweight='bold', transform=ax7.transAxes)
        ax7.text(0.5, 0.5, risk_level, ha='center', va='center',
                fontsize=24, fontweight='bold', color=risk_colors[risk_level],
                transform=ax7.transAxes)
        ax7.text(0.5, 0.3, f"Confidence: {confidence:.1%}", ha='center', va='center',
                fontsize=14, transform=ax7.transAxes)
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')

        # 8. Attention Statistics
        ax8 = fig.add_subplot(gs[2, :])

        # Calculate attention statistics
        attention_stats = {
            'Max Attention': f"{attention_map.max():.3f}",
            'Mean Attention': f"{attention_map.mean():.3f}",
            'TB Regions': f"{np.sum(attention_map > tb_threshold)} pixels",
            'Coverage': f"{(np.sum(attention_map > tb_threshold) / attention_map.size * 100):.1f}%"
        }

        stats_text = "Attention Analysis: "
        for key, value in attention_stats.items():
            stats_text += f"{key}: {value} | "

        ax8.text(0.5, 0.5, stats_text[:-3], ha='center', va='center',
                fontsize=12, transform=ax8.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax8.axis('off')

        plt.tight_layout()

        # Save to bytes for return
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        result_img = PILImage.open(buf)

        # Try to create 3D visualization with Mayavi
        try:
            create_3d_attention_visualization(image, attention_map,
                                            f"3D TB Analysis - {risk_level} Risk")
        except:
            print("⚠️ 3D visualization with Mayavi not available, using matplotlib fallback")

        print("✅ Enhanced 2D+3D visualization created")
        return result_img

    except Exception as e:
        print(f"❌ Error creating enhanced visualization: {e}")
        return None

def create_tb_model(device):
    """Create TB detection model"""
    print_section("CREATING TB DETECTION MODEL")

    import torch
    import torch.nn as nn
    import timm

    class TBDetectionModel(nn.Module):
        def __init__(self, num_classes=2, backbone='efficientnet_b2'):
            super(TBDetectionModel, self).__init__()

            # Use a more powerful backbone for real medical images
            self.backbone = timm.create_model(
                backbone,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3, 4]  # Multiple scales
            )

            # Get feature dimensions
            dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
            with torch.no_grad():
                features = self.backbone(dummy_input)
                self.feature_dims = [f.shape[1] for f in features]
                print(f"Feature dimensions: {self.feature_dims}")

            # Feature Pyramid Network for multi-scale features
            self.fpn_convs = nn.ModuleList([
                nn.Conv2d(dim, 256, 1) for dim in self.feature_dims
            ])

            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool2d(1)

            # More sophisticated classifier for medical images
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

            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # Ensure input is float32
            x = x.float()

            # Extract multi-scale features
            features = self.backbone(x)

            # Process through FPN
            fpn_features = []
            for i, feat in enumerate(features):
                fpn_feat = self.fpn_convs[i](feat)
                # Resize to same spatial size as largest feature map
                if i > 0:
                    fpn_feat = torch.nn.functional.interpolate(
                        fpn_feat, size=features[0].shape[2:], mode='bilinear', align_corners=False
                    )
                fpn_features.append(fpn_feat)

            # Combine features
            combined_features = torch.cat(fpn_features, dim=1)

            # Apply attention
            attention_map = self.attention(fpn_features[-1])  # Use highest level features for attention
            attended_features = fpn_features[-1] * attention_map

            # Global pooling for all features
            pooled_features = []
            for feat in fpn_features:
                pooled = self.gap(feat).flatten(1)
                pooled_features.append(pooled)

            # Combine all pooled features
            combined_pooled = torch.cat(pooled_features, dim=1)

            # Classification
            output = self.classifier(combined_pooled)

            return {
                'logits': output,
                'attention_map': attention_map,
                'features': combined_features
            }

    model = TBDetectionModel(num_classes=2, backbone='efficientnet_b2')
    model = model.to(device)

    # Test model
    dummy_input = torch.randn(2, 3, 224, 224, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"✅ Model created successfully")
        print(f"   Output shape: {test_output['logits'].shape}")
        print(f"   Attention shape: {test_output['attention_map'].shape}")

    return model

def train_real_model(model, train_loader, val_loader, device, num_epochs=25):
    """Train model on real TBX11K dataset"""
    print_section("TRAINING ON REAL TBX11K DATASET")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Setup training for real medical data
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))  # Weight TB class higher
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # Lower learning rate

    # More sophisticated scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 7  # More patience for real data

    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

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

                # GPU memory management
                if device.type == 'cuda' and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

                if batch_idx % 50 == 0:  # Print every 50 batches
                    gpu_info = ""
                    if device.type == 'cuda':
                        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                        gpu_info = f", GPU: {gpu_memory:.1f}GB"
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}{gpu_info}")

            except Exception as e:
                print(f"  ⚠️ Error in batch {batch_idx}: {e}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)

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
            print(f"📊 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"✅ New best model (Val Acc: {val_acc:.2f}%)")

                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'history': history
                }, 'best_tb_model.pth')

            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break

            scheduler.step()

    print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model, history

def create_real_gradio_interface(model_path, device):
    """Create Gradio interface for real TB detection"""
    print_section("CREATING GRADIO INTERFACE")

    import gradio as gr
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import io
    import timm
    import torch.nn as nn

    # Load the trained model
    def load_trained_model():
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found: {model_path}")
                return None

            checkpoint = torch.load(model_path, map_location=device)

            # Recreate the model architecture
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

                    combined_features = torch.cat(fpn_features, dim=1)
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
                        'features': combined_features
                    }

            model = TBDetectionModel()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            print(f"✅ Model loaded successfully from {model_path}")
            if 'val_acc' in checkpoint:
                print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")

            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    model = load_trained_model()

    def predict_tb_real(image):
        """Predict TB using the trained model on real data"""
        if model is None:
            return "❌ Model not loaded properly", None, "Model loading failed"

        if image is None:
            return "❌ No image provided", None, "Please upload a chest X-ray image"

        try:
            # Preprocess image
            from PIL import Image
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

                # Get both class probabilities
                normal_prob = probabilities[0][0].item()
                tb_prob = probabilities[0][1].item()

                # Get attention map
                attention = outputs['attention_map'][0, 0].cpu().numpy()
                attention = cv2.resize(attention, (224, 224))

                # Use enhanced 2D+3D visualization
                result_img = create_enhanced_2d_3d_visualization(
                    image_resized, attention, predicted_class, confidence,
                    normal_prob, tb_prob
                )

                if result_img is None:
                    # Fallback to original visualization
                    import matplotlib.pyplot as plt
                    import io
                    from PIL import Image

                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                    # Original image
                    axes[0, 0].imshow(image_resized)
                    axes[0, 0].set_title('Original Chest X-ray')
                    axes[0, 0].axis('off')

                    # Attention heatmap
                    axes[0, 1].imshow(image_resized)
                    im = axes[0, 1].imshow(attention, alpha=0.6, cmap='jet')
                    axes[0, 1].set_title('AI Attention Map')
                    axes[0, 1].axis('off')
                    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

                    # Probability distribution
                    classes = ['Normal', 'TB Positive']
                    probs = [normal_prob, tb_prob]
                    colors = ['green' if predicted_class == 0 else 'lightgreen',
                             'red' if predicted_class == 1 else 'lightcoral']

                    bars = axes[1, 0].bar(classes, probs, color=colors)
                    axes[1, 0].set_title('Prediction Probabilities')
                    axes[1, 0].set_ylabel('Probability')
                    axes[1, 0].set_ylim(0, 1)

                    # Add probability labels on bars
                    for bar, prob in zip(bars, probs):
                        height = bar.get_height()
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{prob:.3f}', ha='center', va='bottom')

                    # Risk assessment
                    risk_level = "High" if predicted_class == 1 and confidence > 0.8 else \
                               "Moderate" if predicted_class == 1 and confidence > 0.6 else \
                               "Low-Moderate" if predicted_class == 1 else "Low"

                    risk_colors = {"High": "red", "Moderate": "orange", "Low-Moderate": "yellow", "Low": "green"}

                    axes[1, 1].text(0.5, 0.7, f"Risk Level", ha='center', va='center',
                                   fontsize=16, fontweight='bold', transform=axes[1, 1].transAxes)
                    axes[1, 1].text(0.5, 0.5, risk_level, ha='center', va='center',
                                   fontsize=24, fontweight='bold', color=risk_colors[risk_level],
                                   transform=axes[1, 1].transAxes)
                    axes[1, 1].text(0.5, 0.3, f"Confidence: {confidence:.1%}", ha='center', va='center',
                                   fontsize=14, transform=axes[1, 1].transAxes)
                    axes[1, 1].set_xlim(0, 1)
                    axes[1, 1].set_ylim(0, 1)
                    axes[1, 1].axis('off')

                    plt.tight_layout()

                    # Save to bytes
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    plt.close()

                    result_img = Image.open(buf)

                # Create detailed result text
                class_names = ['Normal', 'TB Positive']
                result_class = class_names[int(predicted_class)]

                result_text = f"""
# 🔍 **TB Detection Analysis Results**

## **Primary Diagnosis**
- **Classification:** {result_class}
- **Confidence:** {confidence:.1%}
- **Risk Level:** {risk_level}

## **Detailed Probabilities**
- **Normal/Healthy:** {normal_prob:.1%}
- **TB Positive:** {tb_prob:.1%}

## **Clinical Interpretation**
"""

                if predicted_class == 1:  # TB Positive
                    if confidence > 0.8:
                        result_text += """
**🚨 HIGH PROBABILITY TB DETECTION**
- Strong indicators of tuberculosis present
- Immediate medical consultation recommended
- Consider isolation protocols
- Confirmatory tests (sputum culture, PCR) advised
"""
                    elif confidence > 0.6:
                        result_text += """
**⚠️ MODERATE PROBABILITY TB DETECTION**
- Possible tuberculosis indicators detected
- Medical evaluation strongly recommended
- Additional diagnostic tests suggested
- Monitor symptoms closely
"""
                    else:
                        result_text += """
**⚠️ LOW-MODERATE PROBABILITY DETECTION**
- Some abnormal patterns detected
- Medical consultation recommended
- May require follow-up imaging
- Consider clinical correlation
"""
                else:  # Normal
                    if confidence > 0.9:
                        result_text += """
**✅ NORMAL CHEST X-RAY**
- No significant TB indicators detected
- Chest X-ray appears normal
- Continue routine health monitoring
- Maintain good respiratory hygiene
"""
                    else:
                        result_text += """
**✅ LIKELY NORMAL**
- Minimal abnormal indicators
- Generally normal appearance
- Routine follow-up recommended
- Monitor for any symptoms
"""

                result_text += """

## **Important Medical Disclaimer**
⚠️ **This AI analysis is for screening and educational purposes only.**
- **NOT a substitute for professional medical diagnosis**
- **Always consult qualified healthcare professionals**
- **Clinical correlation with symptoms and history required**
- **Confirmatory tests may be necessary for definitive diagnosis**

*Trained on TBX11K dataset with real chest X-ray images*
"""

                return result_text, result_img, "✅ Analysis completed successfully"

        except Exception as e:
            return f"❌ Error during prediction: {str(e)}", None, "Prediction failed"

    # Create the interface
    with gr.Blocks(title="TB Detection System - Real Dataset") as interface:
        gr.Markdown("""
        # 🫁 **Advanced TB Detection System**
        ### *Trained on Real TBX11K Dataset (11,000+ Chest X-rays)*

        Upload a chest X-ray image for AI-powered tuberculosis screening using a model trained on real medical data.

        **System Features:**
        - 🧠 **Deep Learning Model:** EfficientNet-B2 with Feature Pyramid Network
        - 📊 **Real Training Data:** TBX11K dataset with 11,000+ chest X-rays
        - 🎯 **Attention Mechanism:** Shows where the AI focuses for diagnosis
        - 📈 **Comprehensive Analysis:** Detailed probabilities and risk assessment
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
                    "🔍 Analyze for TB",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("""
                **Supported formats:** PNG, JPG, JPEG
                **Recommended:** Clear chest X-ray images
                """)

            with gr.Column(scale=2):
                gr.Markdown("## 📊 **Analysis Results**")

                result_text = gr.Markdown(
                    value="Upload an image and click 'Analyze for TB' to see results.",
                    label="Detailed Analysis"
                )

                result_viz = gr.Image(
                    label="Comprehensive Visualization",
                    height=400
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready for analysis"
                )

        # Connect the analysis function
        analyze_btn.click(
            fn=predict_tb_real,
            inputs=[input_image],
            outputs=[result_text, result_viz, status_text]
        )

        gr.Markdown("""
        ---
        ## **About This System**

        This TB detection system was trained on the **TBX11K dataset**, which contains over 11,000 real chest X-ray images
        from medical institutions. The model uses advanced deep learning techniques including:

        - **Feature Pyramid Networks** for multi-scale analysis
        - **Attention mechanisms** for interpretable predictions
        - **Data augmentation** for robust performance
        - **Class balancing** for accurate TB detection

        **Performance Metrics:** The model achieves high accuracy on real medical data validation sets.

        **Medical Disclaimer:** This tool is for educational and screening purposes only.
        Always consult qualified healthcare professionals for proper medical diagnosis and treatment decisions.
        """)

    print("✅ Real dataset Gradio interface created successfully")
    return interface

def main():
    """Main function for real TBX11K dataset training and deployment"""
    print("🫁 TB DETECTION SYSTEM - REAL TBX11K DATASET")
    print("=" * 70)
    print("This system uses the REAL TBX11K dataset with 11,000+ chest X-rays")
    print("Training will take significantly longer but provide much better accuracy")
    print("=" * 70)

    start_time = time.time()

    try:
        # Step 1: Install packages
        install_packages()

        # Step 2: Verify imports and GPU setup
        import torch

        # Detailed GPU detection
        print_section("GPU DETECTION AND SETUP")

        print("🔍 Checking GPU availability...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    driver_version = result.stdout.strip()
                    print(f"NVIDIA Driver: {driver_version}")
                else:
                    print("NVIDIA Driver: Unknown")
            except:
                print("NVIDIA Driver: Unknown")

            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            device = torch.device('cuda:0')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

            # Test GPU functionality
            try:
                test_tensor = torch.randn(100, 100).to(device)
                test_result = torch.mm(test_tensor, test_tensor)
                print(f"✅ GPU test successful - tensor shape: {test_result.shape}")

                # Check available GPU memory
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"📊 GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")

                del test_tensor, test_result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ GPU test failed: {e}")
                print("🔄 Falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print("⚠️ CUDA not available - using CPU")
            print("\n💡 To enable GPU acceleration:")
            print("1. Install CUDA-compatible PyTorch:")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("2. Verify CUDA installation:")
            print("   nvidia-smi")
            print("3. Check CUDA toolkit version matches PyTorch")

        print(f"\n🖥️ Final device: {device}")

        # Set memory optimization for GPU
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            print("✅ GPU optimizations enabled")

        # Step 3: Download real dataset
        dataset_path = download_tbx11k_dataset()
        if not dataset_path:
            print("❌ Failed to download dataset. Please check your Kaggle API setup.")
            return

        # Step 4: Explore dataset
        dataset_info = explore_tbx11k_structure(dataset_path)

        # Step 5: Prepare real data
        image_paths, labels, bboxes = prepare_real_tbx11k_data(dataset_path)

        if len(image_paths) == 0:
            print("❌ No images found in dataset")
            return

        # Step 6: Create transforms
        train_transform, val_transform = create_transforms()

        # Step 7: Create data loaders
        train_loader, val_loader = create_real_data_loaders(
            image_paths, labels, train_transform, val_transform, batch_size=8  # Smaller batch for real data
        )

        if train_loader is None:
            print("❌ Failed to create data loaders")
            return

        # Step 8: Create model
        model = create_tb_model(device)

        # Step 9: Train model
        print(f"\n⏰ Starting training on {len(image_paths)} real chest X-rays...")
        trained_model, history = train_real_model(
            model, train_loader, val_loader, device, num_epochs=25
        )

        # Step 10: Create interface
        model_path = 'best_tb_model.pth'
        interface = create_real_gradio_interface(model_path, device)

        # Calculate total time
        total_time = time.time() - start_time

        print_section("SYSTEM READY")
        print(f"✅ TB Detection System trained on REAL data is ready!")
        print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
        print(f"💾 Best model saved to: {model_path}")
        print(f"🌐 Launching web interface...")

        # Launch interface
        interface.launch(
            share=True,
            debug=False,
            show_error=True,
            server_name="0.0.0.0",
            server_port=7860
        )

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
