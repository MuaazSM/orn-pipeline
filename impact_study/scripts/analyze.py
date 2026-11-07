"""Exploratory analysis of the IMPACT figshare release (no training).

Reads NIfTI with numpy only. Inputs (under raw/):
  mandible.zip                      CT image + mandible sub-volume mask per patient
  maxilla/<id>/rs_mask_<id>.nii.gz  maxilla sub-volume masks
  teeth/<id>/rs_mask_<id>.nii.gz    individual teeth masks (ADA 1-32)
  dose_test_cases/<id>/RTDOSE_<id>.nii.gz
Outputs: results.json, figdata.npz
"""
import gzip, io, json, math, os, struct, sys, zipfile
import numpy as np
from scipy import ndimage

DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
      256: np.int8, 512: np.uint16, 768: np.uint32}

TEETH_GROUPS = {
    "Upper central": [6, 7, 8, 9, 10, 11], "Upper premolars": [4, 5, 12, 13],
    "Upper molars": [1, 2, 3, 14, 15, 16], "Lower central": [22, 23, 24, 25, 26, 27],
    "Lower premolars": [20, 21, 28, 29], "Lower molars": [17, 18, 19, 30, 31, 32],
}
ROI, STRIDE = 96, 72  # MONAI sliding window: roi 96^3, default overlap 0.25


def read_nii(blob):
    raw = gzip.decompress(blob)
    e = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    h = raw[:348]
    dim = struct.unpack(e + "8h", h[40:56])
    dt = struct.unpack(e + "h", h[70:72])[0]
    pix = struct.unpack(e + "8f", h[76:108])
    off = int(struct.unpack(e + "f", h[108:112])[0])
    slope, inter = struct.unpack(e + "2f", h[112:120])
    qcode, scode = struct.unpack(e + "2h", h[252:256])
    shape = dim[1:1 + dim[0]][:3]
    arr = np.frombuffer(raw, dtype=np.dtype(DT[dt]).newbyteorder(e),
                        count=int(np.prod(shape)), offset=off).reshape(shape, order="F")
    if scode > 0:
        A = np.vstack([np.array(struct.unpack(e + "12f", h[280:328])).reshape(3, 4), [0, 0, 0, 1]])
    else:
        b, c, d = struct.unpack(e + "3f", h[256:268])
        qx, qy, qz = struct.unpack(e + "3f", h[268:280])
        a = math.sqrt(max(0.0, 1 - b * b - c * c - d * d))
        R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                      [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-c*c-b*b]])
        qfac = -1.0 if pix[0] < 0 else 1.0
        A = np.eye(4)
        A[:3, :3] = R @ np.diag([pix[1], pix[2], pix[3] * qfac])
        A[:3, 3] = [qx, qy, qz]
    if slope not in (0.0, 1.0) or inter != 0.0:
        arr = arr.astype(np.float32) * (slope if slope else 1.0) + inter
    return arr, A


def spacing(A):
    return np.linalg.norm(A[:3, :3], axis=0)


def n_windows(L):
    return 1 if L <= ROI else math.ceil((L - ROI) / STRIDE) + 1


def body_bbox_mm(img, sp):
    """Approximates the repo's GnerateExternalMask(threshold=0) + CropForeground."""
    sub = img[::2, ::2, :] > 0
    lab, n = ndimage.label(sub)
    if n == 0:
        return [img.shape[i] * sp[i] for i in range(3)]
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    idx = np.argwhere(lab == big)
    ext = idx.max(0) - idx.min(0) + 1
    return [float(ext[0] * 2 * sp[0]), float(ext[1] * 2 * sp[1]), float(ext[2] * sp[2])]


def class_stats(mask, img, A, labels, vox_cc):
    out = {}
    for k in labels:
        m = mask == k
        n = int(m.sum())
        if n == 0:
            out[k] = {"present": False, "cc": 0.0}
            continue
        ijk = np.argwhere(m).mean(0)
        xw = float((A @ np.r_[ijk, 1])[0])
        hu = img[m]
        out[k] = {"present": True, "cc": n * vox_cc, "hu_mean": float(hu.mean()),
                  "hu_median": float(np.median(hu)), "x_world": xw}
    return out


def dose_at(mask_bool, A_mask, dose, A_dose):
    ijk = np.argwhere(mask_bool).T.astype(np.float64)
    if ijk.shape[1] == 0:
        return None
    world = A_mask @ np.vstack([ijk, np.ones(ijk.shape[1])])
    didx = (np.linalg.inv(A_dose) @ world)[:3]
    vals = ndimage.map_coordinates(dose.astype(np.float32), didx, order=1, cval=np.nan)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return {"dmean": float(vals.mean()), "dmax": float(vals.max()), "n": int(vals.size)}


def main():
    z = zipfile.ZipFile("raw/mandible.zip")
    names = {i.filename for i in z.infolist()}
    ids = sorted({n.split("/")[1] for n in names if n.count("/") >= 2 and n.endswith(".nii.gz")}, key=int)
    dose_ids = sorted(os.listdir("raw/dose_test_cases"), key=int)
    patients, fig, dose_rows = [], {}, []
    for pid in ids:
        img, A = read_nii(z.read(f"mandible/{pid}/image_{pid}.nii.gz"))
        mand, Am = read_nii(z.read(f"mandible/{pid}/rs_mask_{pid}.nii.gz"))
        maxi, Ax = read_nii(open(f"raw/maxilla/{pid}/rs_mask_{pid}.nii.gz", "rb").read())
        teeth, At = read_nii(open(f"raw/teeth/{pid}/rs_mask_{pid}.nii.gz", "rb").read())
        sp = spacing(A)
        vox_cc = float(np.prod(sp)) / 1000.0
        same_grid = all(m.shape == img.shape and np.allclose(Mx, A, atol=1e-2)
                        for m, Mx in [(mand, Am), (maxi, Ax), (teeth, At)])
        img = img.astype(np.int16) if img.dtype != np.int16 else img
        bb = body_bbox_mm(img, sp)
        crop_vox = [bb[0] / 1.0, bb[1] / 1.0, bb[2] / 2.0]  # after 1x1x2 mm resample
        jaw = (mand > 0) | (maxi > 0) | (teeth > 0)
        if jaw.any():
            lo, hi = np.argwhere(jaw).min(0), np.argwhere(jaw).max(0) + 1
            jaw_img = img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            jaw_gt1600 = float((jaw_img > 1600).mean())
            jaw_max = int(jaw_img.max())
            metal_vox = int((jaw_img > 3000).sum())
        else:
            jaw_gt1600, jaw_max, metal_vox = None, None, None
        p = {
            "id": pid, "shape": list(img.shape), "spacing": [round(float(s), 3) for s in sp],
            "z_extent_mm": float(img.shape[2] * sp[2]), "same_grid": bool(same_grid),
            "body_bbox_mm": bb, "crop_vox": crop_vox,
            "cache_mb": float(np.prod(crop_vox) * 4 * 2 / 1e6),
            "val_windows": int(np.prod([n_windows(L) for L in crop_vox])),
            "hu_p01": float(np.percentile(img[::4, ::4, ::2], 1)),
            "hu_p99": float(np.percentile(img[::4, ::4, ::2], 99)),
            "img_max": int(img.max()), "jaw_frac_gt1600": jaw_gt1600, "jaw_max": jaw_max,
            "jaw_metal_vox_gt3000": metal_vox,
            "mand_labels": sorted(int(v) for v in np.unique(mand) if v),
            "maxi_labels": sorted(int(v) for v in np.unique(maxi) if v),
            "mandible": class_stats(mand, img, A, range(1, 7), vox_cc),
            "maxilla": class_stats(maxi, img, A, range(1, 7), vox_cc),
            "teeth_present": [int(((teeth == k).any())) for k in range(1, 33)],
            "teeth_cc": [float((teeth == k).sum() * vox_cc) for k in range(1, 33)],
            "teeth_labels_max": int(teeth.max()),
        }
        patients.append(p)

        if pid in dose_ids:
            dose, Ad = read_nii(open(f"raw/dose_test_cases/{pid}/RTDOSE_{pid}.nii.gz", "rb").read())
            row = {"id": pid, "dose_shape": list(dose.shape), "dose_spacing": [round(float(s), 2) for s in spacing(Ad)],
                   "dose_max_raw": float(dose.max()), "mandible": {}, "maxilla": {}, "teeth": {}}
            for k in range(1, 7):
                row["mandible"][k] = dose_at(mand == k, A, dose, Ad)
                row["maxilla"][k] = dose_at(maxi == k, A, dose, Ad)
            for g, labs in TEETH_GROUPS.items():
                row["teeth"][g] = dose_at(np.isin(teeth, labs), A, dose, Ad)
            dose_rows.append(row)

        # Figure material: one axial slice through the alveolar mandible + label projections
        if mand.any():
            zc = int(np.argmax((mand >= 4).sum((0, 1))))
            zt = int(np.argmax((teeth > 0).sum((0, 1)))) if teeth.any() else zc
            fig[f"{pid}_axial_img"] = img[:, :, zc].astype(np.int16)
            fig[f"{pid}_axial_mand"] = mand[:, :, zc].astype(np.uint8)
            fig[f"{pid}_axial_teeth_img"] = img[:, :, zt].astype(np.int16)
            fig[f"{pid}_axial_teeth"] = teeth[:, :, zt].astype(np.uint8)
            # coronal "first label hit" projection along y (anterior -> posterior)
            for nm, m in [("mand", mand), ("maxi", maxi), ("teeth", teeth)]:
                hit = (m > 0)
                first = np.argmax(hit, axis=1)
                proj = np.take_along_axis(m, first[:, None, :], axis=1)[:, 0, :]
                proj[~hit.any(1)] = 0
                fig[f"{pid}_cor_{nm}"] = proj.astype(np.uint8)
            fig[f"{pid}_spacing"] = np.array(sp)
            fig[f"{pid}_affine"] = A
        print(pid, img.shape, [round(s, 2) for s in sp], "grid_ok" if same_grid else "GRID_MISMATCH",
              f"teeth={sum(p['teeth_present'])}", flush=True)

    json.dump({"patients": patients, "dose": dose_rows}, open("results.json", "w"), indent=1)
    np.savez_compressed("figdata.npz", **fig)
    print("ANALYZE_DONE", len(patients), "patients,", len(dose_rows), "dose cases")


if __name__ == "__main__":
    main()
