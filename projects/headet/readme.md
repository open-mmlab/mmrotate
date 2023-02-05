# heading detection

本项目尝试支持360的矩形框检测。

# 数据集准备

本仓库在 [ICDAR2019_cTDaRA Modern](https://github.com/cndplab-founder/ICDAR2019_cTDaR)数据集基础上生成而来。


## 数据集下载

```shell
# git clone https://github.com/vansin/ICDAR2019_cTDaR.git -b new ICDAR2019_cTDaR_TRACKA_Modern_HOQ_BBox
git clone https://github.com/vansin/ICDAR2019_cTDaR.git -b new ICDAR2019_MTD_HOQ
```

## 数据集介绍

```shell
ICDAR2019_MTD_HOQ
├── ann_test_hbbox  # 原始HBB水平框测试集标注
├── ann_test_obbox  # 旋转变换后有向边界框测试集标注（算法生成）
├── ann_test_qbbox  # 投影变换后边界框测试集标注（算法生成）
├── ann_train_hbb   # 原始HBB水平框训练集标注
├── ann_train_obbox # 旋转变换后有向边界框训练集标注（算法生成）
├── ann_train_qbbox # 投影变换后边界框训练集标注（算法生成）
├── img_test_hbbox  # 原始测试集图像
├── img_test_obbox  # 旋转变换后测试集图像
├── img_test_qbbox  # 投影变换后测试集图像
├── img_train_hbbox # 原始训练集图片
├── img_train_obbox # 旋转变换后训练集图像
└── img_train_qbbox # 投影变换后训练集图像
```

### hbbox水平框原始数据集

![image](https://user-images.githubusercontent.com/25839884/214336065-7aa155b3-75ca-4e46-85f1-22c47a79de4e.png)

600 个训练图片，240个测试图片

### obbox有向边界框数据集

在 hbbox 水平框原始数据集的基础上，通过随机旋转变换生成 rbbox 旋转边界框数据集，让后通过手动调整部分标注，生成 obbox 有向边界框数据集。
![image](https://user-images.githubusercontent.com/25839884/214334546-b9a940e3-9e88-47ae-aa96-5f2d6444c20c.png)


数据集有由语义信息的，ABCD点是表格左上角开始，依次顺时钟的四个点。
![](https://cdn.vansin.top//picgo/3e4a042cd4b4725c4ae05aa7471467e.png)


### qbbox投影变换后qbbox四边形边界框数据集

在 hbbox 水平框原始数据集的基础上，通过随机投影变换生成 qbbox 四边形边界框数据集。

![image](https://user-images.githubusercontent.com/25839884/214337078-854e530d-6cfb-4e33-82a3-d17c51af39c8.png)




## 数据集config说明

训练集格式为RBBox，测试集格式也为RBBox
project/headet/configs/_base_/datasets/ic19-rbb-rbb.py

训练集格式为OBBox，测试集格式也为OBBox
project/headet/configs/_base_/datasets/ic19-obb-obb.py

训练集格式为HBBox，测试集格式也为OBBox
project/headet/configs/_base_/datasets/ic19-hbb-obb.py

训练集格式为QBBox，测试集格式也为QBBox
project/headet/configs/_base_/datasets/ic19-qbb-qbb.py


## 数据集可视化

```shell
python projects/headet/tools/browse_dataset.py configs/gliding_vertex/gliding-vertex-qbox_r50_fpn_1x_dota.py --stage test
python projects/headet/tools/browse_dataset.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py --stage train
```



# 调试

```shell
python -m debugpy --wait-for-client --listen 5678 tools/train.py projects/headet/configs/rotated_retinanet/rotated-retinanet-hbox-le90_r50_fpn_1x_dota.py
python -m debugpy --wait-for-client --listen 5678 tools/train.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py
```

```shell
 python -m debugpy --wait-for-client --listen 5678 projects/headet/tools/browse_dataset.py projects/headet/configs/gliding_vertex/gliding-vertex-qbox_r50_fpn_1x_dota.py
 python -m debugpy --wait-for-client --listen 5678 projects/headet/tools/browse_dataset.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py --stage train
 ```

 ## RBB 结果

 |config_file|FPS|max_epochs|max_memory|model_group|model_name|test/dota/AP50|
|:----|:----|:----|:----|:----|:----|:----|
|rotated_rtmdet_m-3x-dota_pt.py|21.34 |36|372|rotated_rtmdet|RTMDet|0.898|
|roi-trans-le90_swin-tiny_fpn_3x_dota.py|5.65 |36|608|roi_trans_3x|CascadeRCNN|0.895|
|rotated_rtmdet_tiny-3x-dota_pt.py|52.41 |36|133|rotated_rtmdet|RTMDet|0.891|
|roi-trans-le90_swin-tiny_fpn_3x_dota_pt.py|5.77 |36|609|roi_trans_3x|CascadeRCNN|0.887|
|rotated_rtmdet_l-3x-dota_pt.py|12.73 |36|637|rotated_rtmdet|RTMDet|0.887|
|roi-trans-le90_r50_fpn_3x_dota_pt.py|6.89 |36|598|roi_trans_3x|CascadeRCNN|0.879|
|rotated_rtmdet_s-3x-dota_pt.py|39.64 |36|191|rotated_rtmdet|RTMDet|0.871|
|roi-trans-le90_swin-tiny_fpn_1x_dota.py|5.46 |12|606|roi_trans|CascadeRCNN|0.808|
|roi-trans-le90_r50_fpn_amp-3x_dota_pt.py|6.79 |36|598|roi_trans_3x|CascadeRCNN|0.801|
|roi-trans-le135_r50_fpn_3x_dota.py|7.67 |36|599|roi_trans_3x|CascadeRCNN|0.799|
|roi-trans-le90_r50_fpn_amp-3x_dota.py|8.03 |36|597|roi_trans_3x|CascadeRCNN|0.797|
|roi-trans-le90_r50_fpn_3x_dota.py|7.56 |36|597|roi_trans_3x|CascadeRCNN|0.795|
|roi-trans-le135_r50_fpn_1x_dota.py|7.57 |12|598|roi_trans|CascadeRCNN|0.792|
|roi-trans-le90_r50_fpn_1x_dota.py|7.64 |12|596|roi_trans|CascadeRCNN|0.785|
|roi-trans-oc_r50_fpn_3x_dota.py|8.17 |36|596|roi_trans_3x|CascadeRCNN|0.783|
|rotated-faster-rcnn-le90_r50_fpn_3x_dota_pt.py|7.67 |36|506|rotated_faster_rcnn|FasterRCNN|0.779|
|roi-trans-le90_r50_fpn_amp-1x_dota.py|8.55 |12|596|roi_trans|CascadeRCNN|0.775|
|rotated-faster-rcnn-le90_r50_fpn_3x_dota.py|8.66 |36|506|rotated_faster_rcnn|FasterRCNN|0.761|
|rotated-retinanet-rbox-le90_r50_fpn_3x_dota.py|14.40 |36|361|rotated_retinanet_3x|RetinaNet|0.758|
|rotated-retinanet-rbox-le135_r50_fpn_3x_dota_pt.py|14.33 |36|361|rotated_retinanet_3x|RetinaNet|0.75|
|rotated-retinanet-hbox-oc_r50_fpn_3x_dota.py|14.33 |36|361|rotated_retinanet_3x|RetinaNet|0.746|
|rotated-retinanet-hbox-le90_r50_fpn_3x_dota.py|14.28 |36|361|rotated_retinanet_3x|RetinaNet|0.742|
|rotated-retinanet-hbox-oc_r50_fpn_gwd_1x_dota.py|13.79 |12|363|gwd|RetinaNet|0.728|
|rotated-retinanet-hbox-le135_r50_fpn_3x_dota.py|14.22 |36|361|rotated_retinanet_3x|RetinaNet|0.723|
|rotated-retinanet-rbox-le135_r50_fpn_1x_dota_pt.py|13.94 |12|363|rotated_retinanet|RetinaNet|0.721|
|rotated_rtmdet_m-3x-dota.py|20.34 |36|372|rotated_rtmdet|RTMDet|0.697|
|rotated-retinanet-rbox-le135_r50_fpn_3x_dota.py|14.34 |36|361|rotated_retinanet_3x|RetinaNet|0.683|
|rotated_rtmdet_l-3x-dota.py|12.69 |36|637|rotated_rtmdet|RTMDet|0.679|
|rotated-retinanet-rbox-le90_r50_fpn_amp-3x_dota.py|13.40 |36|361|rotated_retinanet_3x|RetinaNet|0.677|
|rotated-retinanet-rbox-le90_convnext-tiny_fpn_kld-stable_adamw-1x_dota.py|10.63 |12|396|convnext|RetinaNet|0.676|
|roi-trans-oc_r50_fpn_1x_dota.py|7.90 |12|596|roi_trans|CascadeRCNN|0.67|
|rotated_rtmdet_s-3x-dota.py|39.61 |36|191|rotated_rtmdet|RTMDet|0.665|
|rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota.py|13.96 |12|363|gwd|RetinaNet|0.663|
|rotated-fcos-le90_r50_fpn_1x_dota.py|14.33 |12|345|rotated_fcos|FCOS|0.659|
|r3det-refine-oc_r50_fpn_1x_dota.py|7.95 |12|406|r3det|RefineSingleStageDetector|0.654|
|rotated_rtmdet_tiny-3x-dota.py|43.96 |36|133|rotated_rtmdet|RTMDet|0.645|
|rotated-faster-rcnn-le90_r50_fpn_1x_dota.py|8.59 |12|506|rotated_faster_rcnn|FasterRCNN|0.639|
|rotated-retinanet-hbox-le90_r50_fpn_1x_dota.py|13.87 |12|363|rotated_retinanet|RetinaNet|0.637|
|rotated-fcos-hbox-le90_r50_fpn_1x_dota.py|14.39 |12|345|rotated_fcos|FCOS|0.633|
|rotated-retinanet-hbox-le135_r50_fpn_1x_dota.py|14.00 |12|363|rotated_retinanet|RetinaNet|0.606|
|rotated-retinanet-rbox-oc_r50_fpn_3x_dota.py|14.34 |36|361|rotated_retinanet_3x|RetinaNet|0.572|
|rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py|13.96 |12|363|rotated_retinanet|RetinaNet|0.57|
|r3det-oc_r50_fpn_1x_dota.py|10.20 |12|384|r3det|RefineSingleStageDetector|0.555|
|r3det-tiny-oc_r50_fpn_1x_dota.py|12.69 |12|365|r3det|RefineSingleStageDetector|0.528|
|rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py|13.90 |12|363|rotated_retinanet|RetinaNet|0.458|
|rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota.py|14.04 |12|347|rotated_fcos|FCOS|0.388|
|rotated-retinanet-hbox-oc_r50_fpn_1x_dota.py|13.59 |12|363|rotated_retinanet|RetinaNet|0.269|
|rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota.py|13.20 |12|366|csl|RetinaNet|0.262|
|rotated-retinanet-rbox-oc_r50_fpn_1x_dota.py|13.45 |12|363|rotated_retinanet|RetinaNet|0.188|
|rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py|14.49 |12|363|rotated_retinanet|RetinaNet|0|
|rotated-fcos-le90_r50_fpn_kld_1x_dota.py|14.80 |12|345|rotated_fcos|FCOS|0|
|cfa-qbox_r50_fpn_40e_dota.py|13.23 |40|389|cfa|RepPointsDetector|0|
|cfa-qbox_r50_fpn_1x_dota.py|13.27 |12|389|cfa|RepPointsDetector|0|