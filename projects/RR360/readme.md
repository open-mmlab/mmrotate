# RR360(360 Rotated Rectangle Detection)

本项目在ICDAR2019_MTD_HOQ上支持360度旋转矩形框目标检测

# 环境配置

```shell
# 假设已经安装mmengine、mmcv 2.x、mmdetection
git clone https://github.com/open-mmlab/mmrotate -b dev-1.x
cd mmrotate 
export MMROTATE_HOME=$(pwd)
pip install -v -e .
```

# 数据集准备

## 数据集下载

```shell
cd $MMROTATE_HOME
mkdir data
git clone https://github.com/vansin/ICDAR2019_MTD_HOQ.git data/ICDAR2019_MTD_HOQ
```

600 个训练图片，240个测试图片

通过 browse_dataset.py 检测数据集的正确性

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/browse_dataset.py \
    projects/RR360/configs360/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_6x_dota.py
```

数据集有由语义信息的，ABCD点是表格左上角开始，依次顺时钟的四个点。
![](https://cdn.vansin.top//picgo/3e4a042cd4b4725c4ae05aa7471467e.png)

# configs360

推理已训练好的模型

```shell
cd $MMROTATE_HOME

wget https://openmmlab.vansin.top/work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/epoch_36.pth

python projects/RR360/tools/test.py \
    projects/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    epoch_36.pth \
    # --show

```

训练

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/train.py \
    projects/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    --d
```

测试自行训练的

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/test.py \
    work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/epoch_36.pth
```


|config_file|model_group|model_name|test/dota/AP50|test/dota/AP50H90|FPS|max_memory|max_epochs|max_iters|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|rotated_rtmdet_l_l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.897|0.896|25.18980848|419|36|32400|
|rotated_rtmdet_l_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.898|0.895|25.12457385|419|36|32400|
|rotated_rtmdet_m_l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.896|0.893|30.62018724|202|36|32400|
|rotated_rtmdet_m_al1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.898|0.889|29.43219076|202|36|32400|
|rotated_rtmdet_s_l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.89|0.887|34.76708714|72|36|32400|
|rotated_rtmdet_m_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.894|0.886|29.92472451|202|36|32400|
|rotated_rtmdet_s_al1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.875|0.805|34.99214249|72|36|32400|
|rotated_rtmdet_tiny_l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.805|0.805|33.83307671|40|36|32400|
|rotated_rtmdet_tiny_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.801|0.799|34.5884871|40|36|32400|
|rotated_rtmdet_tiny_al1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.801|0.798|34.35242392|40|36|32400|
|rotated_rtmdet_s_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.87|0.795|34.03341603|72|36|32400|
|rotated_rtmdet_l_l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.802|0.794|27.51497858|419|36|32400|
|rotated_rtmdet_m_l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.798|0.79|28.53407624|202|36|32400|
|rotated_rtmdet_l_al1-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.788|0.787|24.99243089|419|36|32400|
|rotated_rtmdet_l-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.799|0.786|28.72683885|419|36|32400|
|rotated_rtmdet_s_l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.798|0.78|34.72571773|72|36|32400|
|rotated_rtmdet_l_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.795|0.779|27.62019302|419|36|32400|
|rotated_rtmdet_m_l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.793|0.704|31.1818681|202|36|10800|
|rotated_rtmdet_l_l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.783|0.702|22.56788876|419|36|10800|
|rotated_rtmdet_l_a1l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.766|0.699|27.27286338|419|36|10800|
|rotated_rtmdet_m_al1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.78|0.697|22.95045136|202|36|10800|
|rotated_rtmdet_s_al1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.785|0.695|34.81889286|72|36|32400|
|rotated_rtmdet_m_al1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.702|0.69|31.28286532|202|36|32400|
|rotated_rtmdet_tiny_l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.772|0.689|35.23428897|40|36|32400|
|rotated_rtmdet_l-3x-ic19.py|rotated_rtmdet_x3_r|RTMDet|0.696|0.683|26.95822517|419|36|32400|
|rotated_rtmdet_tiny_al1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.702|0.68|24.18143699|40|36|10800|
|rotated_rtmdet_tiny_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.776|0.676|35.83994276|40|36|32400|
|rotated_rtmdet_tiny_a1l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.762|0.675|22.82166165|40|36|10800|
|rotated_rtmdet_s_a1l1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.773|0.673|33.53111446|72|36|32400|
|rotated_rtmdet_s_al1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.755|0.673|32.36082248|72|36|10800|
|rotated_rtmdet_tiny_al1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.771|0.672|35.6979651|40|36|32400|
|rotated_rtmdet_l_al1-3x-ic19_pt.py|rotated_rtmdet_x3|RTMDet|0.686|0.67|27.38270746|419|36|32400|
|rotated_rtmdet_tiny_l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.69|0.664|24.68802856|40|36|10800|
|rotated_rtmdet_m_a1l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.692|0.663|30.65861447|202|36|10800|
|rotated_rtmdet_s_a1l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.695|0.659|23.62099277|72|36|10800|
|rotated_rtmdet_l-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.695|0.658|22.38194881|419|36|10800|
|rotated_rtmdet_l_al1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.692|0.607|27.49131127|419|36|10800|
|rotated_rtmdet_l-3x-ic19_pt.py|rotated_rtmdet_x3_r|RTMDet|0.689|0.596|26.61555252|419|36|32400|
|rotated_rtmdet_s_l1-3x-ic19_pt.py|rotated_rtmdet|RTMDet|0.674|0.575|34.29938491|72|36|10800|
|rotated_rtmdet_l-3x-ic19.py|rotated_rtmdet_x3|RTMDet|0.279|0.26|26.20734586|419|36|32400|
|rotated_rtmdet_l-3x-ic19.py|rotated_rtmdet|RTMDet|0.184|0.172|26.48888138|419|36|10800|


 # RBB 结果 (180°)



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