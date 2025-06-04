import os
import shutil

def move_output_files():
    source_folder = "new_data"
    destination_folder = "models_out/format/prompt3/deep"

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if "output" in filename:
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved: {filename}")

if __name__ == "__main__":
    move_output_files()
