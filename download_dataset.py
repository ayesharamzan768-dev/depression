from pathlib import Path
import shutil
import kagglehub

DATASET = "parthanimesh/reddit-posts-on-depression-in-indian-society"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Downloading Kaggle dataset...")
source = Path(kagglehub.dataset_download(DATASET))
print(f"Downloaded to cache: {source}")

for file in source.rglob("*"):
    if file.is_file() and file.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}:
        target = DATA_DIR / file.name
        if not target.exists():
            shutil.copy2(file, target)
            print(f"Copied: {file.name}")
        else:
            print(f"Already exists: {file.name}")
print("Done. Keep the original file name unchanged.")
