# Corn 3D point cloud segmentation Models

- Semantic Segmentation: PointNet++
- Instance Segmentation: HAIS



# train

**PointNet++**

```bash
python corn_train_semseg.py --root {data_root} --batch_size 100 --meta {meta_dir_name} --epoch 100
```

**HAIS**
```bash
python train2.py --config config/hais_run1_corn_v2_1_vega_1000_151_bottom_828_s_15.yaml  # HAIS
```


# test

**PointNet++**

```bash
python corn_test_semseg.py --log_dir {logdir} --visual --meta {meta_dir}
```

**HAIS**
```bash
python batch_test3.py --config config/hais_run1_corn_v2_1_vega_1000_151_bottom_828_s_15.yaml  # HAIS
python batch_test3_post_v2.py --config config/hais_run1_corn_v2_1_vega_1000_151_bottom_828_s_15.yaml  # HAIS_DFSP
```

# Results


| Model Name | mIoU   | stem mIoU | leaf mIoU | Download  |
| ---------- |--------|-----------|-----------|-----------|
| PointNet++ | 91.93% | 85.04%    | 98.82%    | model ([Google Drive](https://drive.google.com/file/d/1z1uBRdiG271mMcnbmPJDyX6IYbYM_4zo/view?usp=drive_link) / [百度网盘](https://pan.baidu.com/s/1KbfWoHkkYyp_JxYFxm_ISA?pwd=syau)) |



| Model Name | mAP    | stem mAP | leaf mAP | Download                                                                                                                 | Config                                                                       |
|------------|--------|----------|----------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| HAIS       | 89.57% | 93.32%   | 85.83%   | model ([Google Drive](https://drive.google.com/file/d/1mn2dmBistMJWZF4jcpg-hXHRqQ1Erxbe/view?usp=drive_link) / [百度网盘](https://pan.baidu.com/s/13aGQIx8PyXZ1oBD72m_p-Q?pwd=syau)) | [config](HAIS/config/hais_run1_corn_v2_1_vega_1000_151_bottom_828_s_15.yaml) |
| HAIS_DFSP  | 93.74% | 92.97%   | 94.44%   | above                                                                                                                    | [config](HAIS/config/hais_run1_corn_v2_1_vega_1000_151_bottom_828_s_15.yaml)                  |


# Reference

- [HAIS](https://github.com/hustvl/HAIS)
- [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
