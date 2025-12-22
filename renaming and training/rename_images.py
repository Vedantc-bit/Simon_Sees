import os
from PIL import Image

# Set the path to your training data directory
TRAIN_DATA_PATH = './nyu_depth_v2_data/data/nyu2_train'

def process_and_rename_files(root_path):
 
    if not os.path.exists(root_path):
        print(f"FATAL: Directory not found at '{root_path}'")
        return

    print(f"--- Starting Full Dataset Processing ---")
    print(f"Searching in: {root_path}")
    corrupt_files = []

    for dirpath, _, filenames in os.walk(root_path):
        print(f"\nProcessing directory: {dirpath}")

        if not filenames:
            print("  - Directory is empty.")
            continue

        for filename in filenames:
            # 1. Skip files that are already correctly named
            if filename.startswith(('rgb_', 'depth_')):
                continue

            # 2. Skip hidden system files
            if filename.startswith('.'):
                continue

            file_path = os.path.join(dirpath, filename)

            try:
                # 3. Try to open the image file
                with Image.open(file_path) as img:
                    base_name, extension = os.path.splitext(filename)

                    new_name = ""
                    # 4. Determine the new name based on image mode
                    if img.mode == 'RGB':
                        new_name = f"rgb_{base_name.zfill(4)}{extension}"
                    elif img.mode == 'L': # 'L' is for grayscale images
                        new_name = f"depth_{base_name.zfill(4)}{extension}"
                    else:
                        print(f"  - Skipping {filename} (unhandled mode: {img.mode})")
                        continue

                    new_file_path = os.path.join(dirpath, new_name)

                    # Check if the target file already exists to be safe
                    if os.path.exists(new_file_path):
                        print(f"  - Target '{new_name}' already exists. Skipping rename of '{filename}'.")
                        continue

                # 5. If successful, close the image and rename the file
                # (The 'with' statement handles closing automatically)
                os.rename(file_path, new_file_path)
                print(f"  - Renamed '{filename}' to '{new_name}'")

            except IOError:
                # 6. If Pillow cannot open it, it's likely corrupt or not an image
                print(f"  - ❗️ ERROR: Could not open or identify '{filename}'. The file may be corrupt.")
                corrupt_files.append(file_path)
            except Exception as e:
                print(f"  - ❗️ An unexpected error occurred with {filename}: {e}")

    print("\n--- Processing Complete! ---")
    if corrupt_files:
        print("\nThe following files appear to be corrupt and could not be processed:")
        for f in corrupt_files:
            print(f"  - {f}")
    else:
        print("All files processed successfully!")

if __name__ == '__main__':
    process_and_rename_files(TRAIN_DATA_PATH)