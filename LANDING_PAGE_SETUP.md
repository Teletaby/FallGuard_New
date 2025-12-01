# Landing Page Background Setup

## Quick Start

The FallGuard landing page has been updated with a professional design using a camera surveillance feed as the background.

### Steps to Add Your Background Image:

1. **Prepare Your Image**
   - Get a high-resolution image (1920x1080 or higher recommended)
   - The second image provided (surveillance camera feed) works perfectly
   - Supported formats: JPG, PNG, WebP

2. **Save the Image**
   - Save your image as: `app/static/images/landing-bg.jpg`
   - Example: `C:\Users\rosen\Music\FallGuard_New-main\app\static\images\landing-bg.jpg`

3. **That's It!**
   - Refresh your browser to see the changes
   - No code changes needed - it will automatically use the background image

### Image Recommendations:

- **Resolution**: 1920x1080 or higher (4K recommended)
- **Format**: JPG for best compression
- **Size**: Keep under 2MB for optimal loading
- **Content**: Surveillance footage, office, or home scenes work well

### Fallback:

If no background image is provided, the landing page will display with a professional dark gradient background.

### Technical Details:

The landing page now includes:
- Full-screen background image with responsive design
- Semi-transparent dark overlay (40-50% opacity) for text readability
- Hero section with title, description, and "Learn More" button
- Smooth animations and transitions
- Mobile-responsive layout

The background image is referenced in:
- `app/index.html` (line 59): `style="background-image: url('/static/images/landing-bg.jpg');"`
- The CSS handles responsive sizing automatically

### Troubleshooting:

**Q: Background not showing?**
- A: Check that the image file exists at `app/static/images/landing-bg.jpg`
- A: Make sure the file is a valid image format (JPG, PNG, WebP)
- A: Clear browser cache and hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

**Q: Image loading slowly?**
- A: Compress the image to reduce file size (keep under 2MB)
- A: Use JPG format for best compression
- A: Consider using a tool like TinyJPG to optimize

**Q: Text not readable on background?**
- A: The overlay opacity is fixed at 40% - you can adjust in CSS (`app/static/css/styles.css`)
- A: Line: `background: linear-gradient(135deg, rgba(0, 0, 0, 0.4)...)`
- A: Change `0.4` to higher value for darker overlay (0.5-0.6 recommended)

### File Locations:

```
app/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── images/
│       └── landing-bg.jpg  ← Place your image here
├── index.html
└── ...
```

Enjoy your professional-looking landing page! 🎨
