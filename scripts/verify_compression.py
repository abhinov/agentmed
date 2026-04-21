import os
from pathlib import Path
from PIL import Image

def main():
    images_dir = Path("data/images")
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist.")
        return
        
    # Pick the first .jpg or .png
    target_img = None
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files = list(images_dir.glob(ext))
        # Keep finding across both upper and lower cases
        files.extend(list(images_dir.glob(ext.upper())))
        
        if files:
            target_img = files[0]
            break

    if not target_img:
        print(f"Error: No images found in {images_dir}")
        return

    print(f"Selected target image: {target_img.name}")
    original_size = os.path.getsize(target_img)
    print(f"Original Size: {original_size / 1024:.2f} KB")

    output_path = "debug_compressed_sample.jpg"

    # Exact Compression logic (max dimensions, RGB conversion, quality reduction)
    MAX_DIMENSION = 1024
    COMPRESSION_QUALITY = 85

    with Image.open(target_img) as img:
        # 1. RGB Conversion
        if img.mode != "RGB":
            print("Converting image to RGB mode...")
            img = img.convert("RGB")
            
        # 2. Max Dimensions (Preserving aspect ratio via thumbnail)
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        
        # 3. Quality Reduction & Optimization
        img.save(output_path, "JPEG", optimize=True, quality=COMPRESSION_QUALITY)

    compressed_size = os.path.getsize(output_path)
    
    print("-" * 30)
    print(f"Compression Complete!")
    print(f"Output saved to: {output_path}")
    print(f"Compressed Size: {compressed_size / 1024:.2f} KB")
    
    # Calculate % reduction
    reduction = (1 - (compressed_size / original_size)) * 100
    print(f"Total Byte Reduction: {reduction:.1f}%")

if __name__ == "__main__":
    main()
