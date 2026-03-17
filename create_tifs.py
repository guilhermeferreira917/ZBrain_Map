import os
import h5py
import scipy.sparse as sp
import numpy as np
import tifffile

mat_path = "MaskDatabase.mat"
out_dir = "mask_tifs"
os.makedirs(out_dir, exist_ok=True)

def read_matlab_string(f, ref):
    arr = f[ref][()]
    arr = np.array(arr).squeeze()
    return "".join(chr(int(x)) for x in arr)

with h5py.File(mat_path, "r") as f:
    data = f["MaskDatabase/data"][()]
    ir = f["MaskDatabase/ir"][()]
    jc = f["MaskDatabase/jc"][()]
    width = int(f["width"][()][0, 0])
    height = int(f["height"][()][0, 0])
    zs = int(f["Zs"][()][0, 0])

    refs = f["MaskDatabaseNames"][()]
    names = [read_matlab_string(f, refs[i, 0]) for i in range(refs.shape[0])]

mask_db = sp.csc_matrix((data, ir, jc), shape=(width * height * zs, len(jc) - 1))

for col in range(mask_db.shape[1]):
    flat = mask_db[:, col].toarray().ravel()

    vol = flat.reshape((height, width, zs), order="F")
    vol = np.transpose(vol, (2, 0, 1))
    vol = (vol > 0).astype(np.uint16) * 65535

    out_path = os.path.join(out_dir, f"{col}.tif")
    tifffile.imwrite(out_path, vol)

    if col % 25 == 0 or col == mask_db.shape[1] - 1:
        print(f"Saved {col + 1}/{mask_db.shape[1]}")

print(f"Done. TIFFs saved in: {out_dir}")