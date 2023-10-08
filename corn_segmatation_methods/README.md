# 玉米三维点云分割方法

- 语义分割 PointNet++
- 实例分割 HAIS



# 训练

**PointNet++**

```bash
python corn_train_semseg.py --root {data_root} --batch_size 100 --meta {meta_dir_name} --epoch 100
```

**HAIS**
```bash
python train2.py --config config/hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml  # HAIS
python train3.py --config config/hais_run1_corn_v2_1_vega_1000_151_bottomc_828_imoprove.yaml  # HAIS+
```


# 测试

**PointNet++**

```bash
python corn_test_semseg.py --log_dir {logdir} --visual --meta {meta_dir}
```

**HAIS**
```bash
python batch_test3.py --config config/hais_run1_corn_v0.yaml  # HAIS
python batch_test3_post.py --config config/hais_run1_corn_v0.yaml  # HAIS_DFSP
python batch_test3_improve.py --config config/hais_run1_corn_v0.yaml  # HAIS+
python batch_test3_improve_post.py --config config/hais_run1_corn_v0.yaml  # HAIS+_DFSP
```

# Results


| Model Name | mIoU   | stem mIoU | leaf mIoU | Download  |
| ---------- |--------|-----------|-----------|-----------|
| PointNet++ | 94.66% | 89.98%    | 99.33%    | model ([Google Drive](https://drive.google.com/file/d/1yIWNWB7HUMAEgDKF9AkaSs3ayT1DE__S/view?usp=drive_link) / [百度网盘]()) |



| Model Name | mAP    | stem mAP | leaf mAP | Download                                                                                                                 | Config                                                      |
| ---------- |--------|----------|----------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| HAIS       | 88.99% | 91.99%   | 85.98%   | model ([Google Drive](https://drive.google.com/file/d/1PhzWIkfW20tyPeXan2b-LfUfXFa_ipnt/view?usp=drive_link) / [百度网盘]()) | [config](hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml) |
| HAIS       | 91.52% | 92.37%   | 90.66%   | above                                                                                                                    |                                            [config](hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml)                 |
| HAIS+      | 88.98% | 92.07%   | 85.90%   | model ([Google Drive](https://drive.google.com/file/d/1_3GnnfzBzhbiJQi6h96aebt_fYav_fwK/view?usp=drive_link) / [百度网盘]()) | [config](hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml)                                                  |
| HAIS+_DFSP | 91.80% | 92.29%   | 91.30%   | above                                                                                                                    |                                                [config](hais_run1_corn_v2_1_vega_1000_151_bottom_828.yaml)              |