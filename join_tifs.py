import os
import re
import glob
import numpy as np
import tifffile
import h5py

tif_dir = "transformed_tifs"  
h5_path = "masks_zyxp.h5"

def numeric_prefix_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d+)_", name)
    if not m:
        raise ValueError(f"Filename does not start with '<number>_': {name}")
    return int(m.group(1))

tif_files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")), key=numeric_prefix_key)

if not tif_files:
    raise ValueError(f"No TIFF files found in {tif_dir}")

first = tifffile.imread(tif_files[0])
if first.ndim != 3:
    raise ValueError(f"Expected 3D TIFF, got shape {first.shape} for {tif_files[0]}")

z, y, x = first.shape
p = len(tif_files)

print("TIFF shape (z, y, x):", (z, y, x))
print("Number of channels p:", p)

with h5py.File(h5_path, "w") as f:
    dset = f.create_dataset(
        "masks",
        shape=(z, y, x, p),
        dtype=first.dtype,
        compression="gzip"
    )

    for i, tif_path in enumerate(tif_files):
        vol = tifffile.imread(tif_path)

        if vol.shape != (z, y, x):
            raise ValueError(
                f"Shape mismatch in {tif_path}: got {vol.shape}, expected {(z, y, x)}"
            )

        dset[:, :, :, i] = vol

        if i % 25 == 0 or i == p - 1:
            print(f"Added {i+1}/{p}: {os.path.basename(tif_path)}")

print(f"Done. Saved: {h5_path}")


with h5py.File("masks_zyxp.h5", "r") as f:
    print(f["masks"].shape)