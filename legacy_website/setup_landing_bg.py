#!/usr/bin/env python3
"""
Setup script to configure the FallGuard landing page background image.

Instructions:
1. Replace 'landing-bg.jpg' with your desired background image
2. Place the image file in: app/static/images/landing-bg.jpg
3. Image should be high-resolution (1920x1080 or higher recommended)
4. Supported formats: JPG, PNG, WebP
"""

import os
import sys
from pathlib import Path

def setup_landing_bg():
    """Setup the landing page background image"""
    
    app_dir = Path(__file__).parent / 'app'
    static_dir = app_dir / 'static'
    images_dir = static_dir / 'images'
    
    # Create images directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Images directory ready: {images_dir}")
    
    # Check if background image exists
    bg_image = images_dir / 'landing-bg.jpg'
    
    if bg_image.exists():
        size_mb = bg_image.stat().st_size / (1024 * 1024)
        print(f"✓ Landing background image found: {bg_image}")
        print(f"  File size: {size_mb:.2f} MB")
        return True
    else:
        print(f"\n⚠ Landing background image not found!")
        print(f"  Expected location: {bg_image}")
        print(f"\n📸 To use a background image:")
        print(f"  1. Place your image file at: {bg_image}")
        print(f"  2. Supported formats: JPG, PNG, WebP")
        print(f"  3. Recommended resolution: 1920x1080 or higher")
        print(f"\n💡 If you don't add an image, a fallback gradient will be used.")
        return False

def verify_setup():
    """Verify the setup is complete"""
    
    print("\n" + "="*60)
    print("LANDING PAGE SETUP VERIFICATION")
    print("="*60 + "\n")
    
    # Check HTML
    html_file = Path(__file__).parent / 'app' / 'index.html'
    if html_file.exists():
        content = html_file.read_text()
        if 'landing-bg.jpg' in content:
            print("✓ HTML updated with background image reference")
        else:
            print("⚠ HTML may not have the background image reference")
    
    # Check CSS
    css_file = Path(__file__).parent / 'app' / 'static' / 'css' / 'styles.css'
    if css_file.exists():
        print("✓ CSS file found")
    
    # Check images directory
    images_dir = Path(__file__).parent / 'app' / 'static' / 'images'
    if images_dir.exists():
        print(f"✓ Images directory ready: {images_dir}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    print("\n🎨 FallGuard Landing Page Background Setup\n")
    
    setup_landing_bg()
    verify_setup()
    
    print("\n✅ Setup complete!\n")
