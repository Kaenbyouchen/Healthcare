import os, json, numpy as np, nibabel as nib
root = r"D:\Healthcare\nnunet\nnUNet_raw\Dataset001_Ruishan_T1_VIBE_Data_30"
lbl_dir = os.path.join(root, "labelsTr")
all_values = set()
files_with_11 = []
per_file = []
for fn in sorted(os.listdir(lbl_dir)):
    if not fn.lower().endswith(".nii.gz"):
        continue
    p = os.path.join(lbl_dir, fn)
    img = nib.load(p)
    data = img.get_fdata()
    u, c = np.unique(data, return_counts=True)
    u = [int(x) for x in u]
    per_file.append((fn, dict(zip(u, c.tolist()))))
    all_values.update(u)
    if 11 in u:
        files_with_11.append((fn, int(dict(zip(u, c)).get(11, 0))))
print("=== 全部标签取值 ===")
print(sorted(all_values))
print("\n=== 含有 11 的病例 (文件名, 11的体素数) ===")
for fn, cnt in files_with_11:
    print(fn, cnt)
print(f"\n共有 {len(files_with_11)} 个病例含有 11")
# 额外导出一个简要报告
rep = {
    "all_values_sorted": sorted(all_values),
    "files_with_11": [{"file": f, "voxels_11": cnt} for f, cnt in files_with_11],
    "per_file_counts": per_file
}
with open(os.path.join(root, "label_value_report.json"), "w", encoding="utf-8") as f:
    json.dump(rep, f, indent=2, ensure_ascii=False)
print(f"\n报告已写入: {os.path.join(root, 'label_value_report.json')}")