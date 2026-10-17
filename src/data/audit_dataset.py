from pathlib import Path
from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_FOLDERS = {
    "homogeneous": 0,
    "inhomogeneous": 1,
}

records = []
hashes = defaultdict(list)

for folder in ["homogeneous", "inhomogeneous", "imgs"]:

    image_dir = DATA_DIR / folder

    for image_path in sorted(image_dir.glob("*.png")):

        mask_path = DATA_DIR / "masks" / image_path.name

        record = {
            "image_id": image_path.stem,
            "folder": folder,
            "label": CLASS_FOLDERS.get(folder),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "mask_exists": mask_path.exists(),
        }

        try:
            with Image.open(image_path) as img:

                img.load()

                record["width"] = img.width
                record["height"] = img.height
                record["mode"] = img.mode

                array = np.asarray(img)

                record["mean_intensity"] = float(array.mean())
                record["std_intensity"] = float(array.std())

                # Hash decoded pixel data, not just PNG bytes
                fingerprint = sha256(
                    img.mode.encode()
                    + str(img.size).encode()
                    + array.tobytes()
                ).hexdigest()

                hashes[fingerprint].append(str(image_path))

                record["corrupt"] = False

        except Exception as exc:

            record["corrupt"] = True
            record["error"] = str(exc)

        if mask_path.exists():

            try:
                with Image.open(mask_path) as mask:

                    mask.load()

                    mask_array = np.asarray(mask)

                    record["mask_width"] = mask.width
                    record["mask_height"] = mask.height
                    record["mask_mode"] = mask.mode

                    record["mask_values"] = str(
                        np.unique(mask_array).tolist()
                    )

                    record["dimensions_match"] = (
                        record.get("width") == mask.width
                        and record.get("height") == mask.height
                    )

            except Exception as exc:

                record["mask_error"] = str(exc)

        records.append(record)

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_DIR / "dataset_audit.csv",
    index=False
)

print("\n=== DATASET AUDIT ===")

print("\nTotal images:", len(df))

print("\nFolder distribution:")
print(df["folder"].value_counts())

print("\nImage modes:")
print(df["mode"].value_counts(dropna=False))

print("\nImage dimensions:")
print(
    df.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\nCorrupt images:")
print(df["corrupt"].sum())

print("\nMissing masks:")
print((~df["mask_exists"]).sum())

print("\nImage/mask dimension mismatches:")
print((df["dimensions_match"] == False).sum())

print("\nMask pixel values:")
print(df["mask_values"].value_counts().head(15))

duplicates = {
    key: paths
    for key, paths in hashes.items()
    if len(paths) > 1
}

print("\nExact duplicate image groups:")
print(len(duplicates))

for paths in list(duplicates.values())[:10]:
    print(paths)

print("\nAudit saved to:")
print(OUTPUT_DIR / "dataset_audit.csv")