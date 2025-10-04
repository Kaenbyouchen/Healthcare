import nibabel as nib
img = nib.load(r'D:\USC Research Ruishan Liu\multi-organ segmentation论文整理\Ruishan T1_VIBE_Data_30-20250916T000037Z-1-001\T1_VIBE_Data_30\case0000_img.nii.gz')
print(img.header)
