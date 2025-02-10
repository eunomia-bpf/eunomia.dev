import os
import shutil
from pathlib import Path

def reorganize_files():
    # Define paths
    base_dir = Path("docs/")
    en_dir = base_dir / "en"
    zh_dir = base_dir / "zh"
    
    # Create language directories if they don't exist
    en_dir.mkdir(exist_ok=True, parents=True)
    zh_dir.mkdir(exist_ok=True, parents=True)
    
    def process_directory(current_dir: Path, en_base: Path, zh_base: Path):
        # Create corresponding language subdirectories
        relative_path = current_dir.relative_to(base_dir) if current_dir != base_dir else Path("")
        current_en_dir = en_base / relative_path
        current_zh_dir = zh_base / relative_path
        current_en_dir.mkdir(exist_ok=True, parents=True)
        current_zh_dir.mkdir(exist_ok=True, parents=True)
        
        # Process all files in the current directory
        for item in current_dir.iterdir():
            # Skip the language directories themselves
            if item.is_dir():
                if item.name not in ['en', 'zh']:
                    process_directory(item, en_base, zh_base)
                continue
                
            if item.suffix == '.md':
                filename = item.name
                if filename.endswith(".zh.md"):
                    # Chinese file
                    new_filename = filename.replace(".zh.md", ".md")
                    new_path = current_zh_dir / new_filename
                    if not new_path.exists():
                        shutil.move(str(item), str(new_path))
                else:
                    # English file (no .zh suffix)
                    new_path = current_en_dir / filename
                    if not new_path.exists():
                        shutil.move(str(item), str(new_path))
    
    # Start the recursive process
    process_directory(base_dir, en_dir, zh_dir)
    
    # Move images to a common directory if they exist
    imgs_dir = base_dir / "imgs"
    if imgs_dir.exists():
        new_imgs_dir = base_dir / "../imgs"
        if not new_imgs_dir.exists():
            shutil.move(str(imgs_dir), str(new_imgs_dir))

    print("Files have been reorganized into language-specific directories")

if __name__ == "__main__":
    reorganize_files()
