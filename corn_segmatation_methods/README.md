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
python train2.py --config config/hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml  # HAIS
```


# test

**PointNet++**

```bash
python corn_test_semseg.py --log_dir {logdir} --visual --meta {meta_dir}
```

**HAIS**
```bash
python batch_test3.py --config config/hais_run1_corn_v0.yaml  # HAIS
python batch_test3_post.py --config config/hais_run1_corn_v0.yaml  # HAIS_DFSP
```

# Results


| Model Name | mIoU   | stem mIoU | leaf mIoU | Download  |
| ---------- |--------|-----------|-----------|-----------|
| PointNet++ | 94.66% | 89.98%    | 99.33%    | model ([Google Drive](https://drive.google.com/file/d/1yIWNWB7HUMAEgDKF9AkaSs3ayT1DE__S/view?usp=drive_link) / [百度网盘](https://pan.baidu.com/s/1KbfWoHkkYyp_JxYFxm_ISA?pwd=syau)) |



| Model Name | mAP    | stem mAP | leaf mAP | Download                                                                                                                 | Config                                                      |
|------------|--------|----------|----------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| HAIS       | 88.99% | 91.99%   | 85.98%   | model ([Google Drive](https://drive.google.com/file/d/1PhzWIkfW20tyPeXan2b-LfUfXFa_ipnt/view?usp=drive_link) / [百度网盘](https://pan.baidu.com/s/13aGQIx8PyXZ1oBD72m_p-Q)) | [config](hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml) |
| HAIS_DFSP  | 91.52% | 92.37%   | 90.66%   | above                                                                                                                    |                                            [config](hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml)                 |


# Reference

- [HAIS](https://github.com/hustvl/HAIS)
- [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
