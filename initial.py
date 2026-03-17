import h5py
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

mat_path = "MaskDatabase.mat"   # change only if needed

with h5py.File(mat_path, "r") as f:
    data = f["MaskDatabase/data"][()]
    ir = f["MaskDatabase/ir"][()]
    jc = f["MaskDatabase/jc"][()]
    width = int(f["width"][()][0, 0])
    height = int(f["height"][()][0, 0])
    zs = int(f["Zs"][()][0, 0])

mask_db = sp.csc_matrix((data, ir, jc), shape=(width * height * zs, len(jc) - 1))

flat = mask_db[:, 0].toarray().ravel()

vol1 = flat.reshape((height, width, zs), order="F")
vol2 = flat.reshape((height, width, zs), order="C")
vol3 = flat.reshape((zs, height, width), order="F")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(vol1.max(axis=2), cmap="gray")
axes[0].set_title("Option 1: (height, width, zs), order='F'")
axes[0].axis("off")

axes[1].imshow(vol2.max(axis=2), cmap="gray")
axes[1].set_title("Option 2: (height, width, zs), order='C'")
axes[1].axis("off")

axes[2].imshow(vol3.max(axis=0), cmap="gray")
axes[2].set_title("Option 3: (zs, height, width), order='F'")
axes[2].axis("off")

plt.tight_layout()
plt.show()

print("Dimensions:")
print("width =", width, "height =", height, "zs =", zs)
print("mask_db shape =", mask_db.shape)
print("column 0 nonzero =", np.count_nonzero(flat))