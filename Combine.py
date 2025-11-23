import os
import re
import shutil

SOURCE_DIR = "frames_output"          #folder that contains numbered subfolders
OUTPUT_DIR = "frames_output\combined"      #target folder to hold all images
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  #which file types to include
PRESERVE_EXTENSION = True    #if False, rename all outputs to .jpg
START_INDEX = 1              #first output index (1-based)


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS

def numeric_key(name):

    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else base

def main():
    #Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Collect all image paths from subfolders
    all_images = []
    for entry in os.scandir(SOURCE_DIR):
        if entry.is_dir():
            #list images in this subfolder
            images = [
                os.path.join(entry.path, f.name)
                for f in os.scandir(entry.path)
                if f.is_file() and is_image_file(f.name)
            ]
            #sort numerically within each folder
            images.sort(key=numeric_key)
            all_images.extend(images)

    #Determine padding based on total image count
    total = len(all_images)
    if total == 0:
        print("No images found. Check SOURCE_DIR and IMAGE_EXTS.")
        return
    pad = 0

    print(f"Found {total} images across subfolders. Writing to '{OUTPUT_DIR}'...")

    #Copy and rename sequentially
    idx = START_INDEX
    for src_path in all_images:
        if PRESERVE_EXTENSION:
            ext = os.path.splitext(src_path)[1].lower()
        else:
            ext = ".jpg"

        dst_name = f"{str(idx).zfill(pad)}{ext}"
        dst_path = os.path.join(OUTPUT_DIR, dst_name)

        #If destination exists (e.g., rerun), ensure unique naming by advancing the index
        while os.path.exists(dst_path):
            idx += 1
            dst_name = f"{str(idx).zfill(pad)}{ext}"
            dst_path = os.path.join(OUTPUT_DIR, dst_name)

        #Use copy2 to preserve timestamps/metadata; switch to move if you want to remove originals
        shutil.copy2(src_path, dst_path)
        idx += 1

    print(f"Done. Saved {total} images to '{OUTPUT_DIR}' as {str(START_INDEX).zfill(pad)}..{str(START_INDEX+total-1).zfill(pad)}")

if __name__ == "__main__":
    main()
