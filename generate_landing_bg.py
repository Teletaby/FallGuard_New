#!/usr/bin/env python3
"""
Generate a placeholder landing page background image.
This creates a professional surveillance camera-style background.

Run this if you don't have a background image ready yet.
"""

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    print("❌ Pillow library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageDraw, ImageFilter, ImageFont


def create_surveillance_bg():
    """Create a professional surveillance camera background"""
    
    # Create image with professional surveillance colors
    width, height = 1920, 1080
    
    # Base color - dark with green tint (surveillance/tech feel)
    base_color = (15, 25, 35)  # Very dark blue-gray
    
    # Create image
    img = Image.new('RGB', (width, height), base_color)
    draw = ImageDraw.Draw(img)
    
    # Add subtle grid pattern overlay (surveillance aesthetic)
    grid_spacing = 80
    grid_color = (40, 60, 80)  # Dark blue
    
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Add subtle noise texture
    noise = Image.new('RGB', (width, height))
    noise_data = noise.load()
    
    import random
    for y in range(height):
        for x in range(width):
            noise_val = random.randint(0, 15)
            noise_data[x, y] = (noise_val, noise_val, noise_val)
    
    # Blend noise with main image
    img = Image.blend(img, noise, 0.03)
    
    # Add gradient overlay for depth
    gradient = Image.new('RGB', (width, height), (0, 0, 0))
    gradient_data = gradient.load()
    
    for y in range(height):
        # Create vertical gradient
        alpha = int((y / height) * 40)
        for x in range(width):
            gradient_data[x, y] = (alpha, alpha, alpha)
    
    img = Image.blend(img, gradient, 0.3)
    
    # Add some tech elements - subtle circles/highlights
    tech_color = (50, 150, 200)  # Tech blue
    
    # Top-right glow effect
    for size in range(200, 50, 10):
        transparency = (50 - size // 4) // 5
        draw.ellipse(
            [(width - size, -size//2), (width + size, size)],
            outline=tech_color,
            width=1
        )
    
    # Add slight vignette
    vignette = Image.new('RGB', (width, height), (0, 0, 0))
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=100))
    img = Image.blend(img, vignette, 0.2)
    
    return img


def save_background():
    """Save the background image"""
    
    output_dir = Path(__file__).parent / 'app' / 'static' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'landing-bg.jpg'
    
    print("🎨 Generating professional surveillance-style background...")
    img = create_surveillance_bg()
    
    print(f"💾 Saving to: {output_path}")
    img.save(output_path, 'JPEG', quality=85, optimize=True)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"✅ Background image created successfully!")
    print(f"📊 File size: {size_kb:.1f} KB")
    print(f"📐 Resolution: 1920x1080")
    
    print(f"\n✨ Your landing page is ready to go!")
    print(f"💡 You can replace this with your own image anytime by:")
    print(f"   1. Saving your image to: {output_path}")
    print(f"   2. Refreshing your browser")


if __name__ == '__main__':
    print("\n🚀 FallGuard Landing Page Background Generator\n")
    
    try:
        save_background()
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 If Pillow installation fails, you can manually:")
        print(f"   1. Place your image at: app/static/images/landing-bg.jpg")
        print(f"   2. Supported formats: JPG, PNG, WebP")
        sys.exit(1)
    
    print()
