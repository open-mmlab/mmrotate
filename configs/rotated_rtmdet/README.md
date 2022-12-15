# RTMDet-R

<!-- [ALGORITHM] -->

## Abstract

Our tech-report will be released soon.

<div align=center>
<img src="https://user-images.githubusercontent.com/11705038/204995787-ef739910-e196-42c7-a9db-c9c8e28a494d.jpg" height="360"/>
</div>

## Results and Models

### DOTA-v1.0

|  Backbone   | pretrain |  Aug  | mmAP  | mAP50 | mAP75 | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny |    IN    |  RR   | 47.37 | 75.36 | 50.64 |   4.88    |  20.45   |         4.40         |        [config](./rotated_rtmdet_tiny-3x-dota.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota_20221201_120814.json)                             |
| RTMDet-tiny |    IN    | MS+RR | 53.59 | 79.82 | 58.87 |   4.88    |  20.45   |         4.40         |      [config](./rotated_rtmdet_tiny-3x-dota_ms.py)       |                       [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota_ms/rotated_rtmdet_tiny-3x-dota_ms-f12286ff.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota_ms/rotated_rtmdet_tiny-3x-dota_ms_20221113_201235.log)                        |
|  RTMDet-s   |    IN    |  RR   | 48.16 | 76.93 | 50.59 |   8.86    |  37.62   |         4.86         |         [config](./rotated_rtmdet_s-3x-dota.py)          |                                   [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota-11f6ccf5.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota_20221124_081442.json)                                   |
|  RTMDet-s   |    IN    | MS+RR | 54.43 | 79.98 | 60.07 |   8.86    |  37.62   |         4.86         |        [config](./rotated_rtmdet_s-3x-dota_ms.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota_ms/rotated_rtmdet_s-3x-dota_ms-20ead048.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota_ms/rotated_rtmdet_s-3x-dota_ms_20221113_201055.json)                             |
|  RTMDet-m   |    IN    |  RR   | 50.56 | 78.24 | 54.47 |   24.67   |  99.76   |         7.82         |         [config](./rotated_rtmdet_m-3x-dota.py)          |                                   [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/rotated_rtmdet_m-3x-dota-beeadda6.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/rotated_rtmdet_m-3x-dota_20221122_011234.json)                                   |
|  RTMDet-m   |    IN    | MS+RR | 55.00 | 80.26 | 61.26 |   24.67   |  99.76   |         7.82         |        [config](./rotated_rtmdet_m-3x-dota_ms.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota_ms/rotated_rtmdet_m-3x-dota_ms-c71eb375.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota_ms/rotated_rtmdet_m-3x-dota_ms_20221122_011234.json)                             |
|  RTMDet-l   |    IN    |  RR   | 51.01 | 78.85 | 55.21 |   52.27   |  204.21  |        10.82         |         [config](./rotated_rtmdet_l-3x-dota.py)          |                                   [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota/rotated_rtmdet_l-3x-dota-23992372.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota/rotated_rtmdet_l-3x-dota_20221122_011241.json)                                   |
|  RTMDet-l   |    IN    | MS+RR | 55.52 | 80.54 | 61.47 |   52.27   |  204.21  |        10.82         |        [config](./rotated_rtmdet_l-3x-dota_ms.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms/rotated_rtmdet_l-3x-dota_ms-2738da34.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms/rotated_rtmdet_l-3x-dota_ms_20221122_011241.json)                             |
|  RTMDet-l   |   COCO   | MS+RR | 56.74 | 81.33 | 63.45 |   52.27   |  204.21  |        10.82         | [config](./rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-dota_ms/rotated_rtmdet_l-coco_pretrain-3x-dota_ms-06d248a2.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-dota_ms/rotated_rtmdet_l-coco_pretrain-3x-dota_ms_20221113_202010.json) |

- By default, DOTA-v1.0 dataset trained with 3x schedule and image size 1024\*1024.

### HRSC

|  Backbone   | pretrain | Aug | mAP 07 | mAP 12 | Params(M) | FLOPS(G) |                   Config                   |                                                                                                                                           Download                                                                                                                                           |
| :---------: | :------: | :-: | :----: | :----: | :-------: | :------: | :----------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny |    IN    | RR  |  90.6  |  97.1  |   4.88    |  12.54   | [config](./rotated_rtmdet_tiny-9x-hrsc.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-9x-hrsc/rotated_rtmdet_tiny-9x-hrsc-9f2e3ca6.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-9x-hrsc/rotated_rtmdet_tiny-9x-hrsc_20221125_145920.json) |

- By default, HRSC dataset trained with 9x schedule and image size 800\*800.

### Stronger augmentation

We also provide configs with Mixup, Mosaic and RandomRotate with longer schedule. Training time is less than MS.

DOTA:

| Backbone | pretrain | schedule |       Aug       | mmAP  | mAP50 | mAP75 |                    Config                     |                                                                                                                                                 Download                                                                                                                                                 |
| :------: | :------: | :------: | :-------------: | :---: | :---: | :---: | :-------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-l |    IN    |   100e   | Mixup+Mosaic+RR | 54.59 | 80.16 | 61.16 | [config](./rotated_rtmdet_l-100e-aug-dota.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-100e-aug-dota/rotated_rtmdet_l-100e-aug-dota-bc59fd88.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-100e-aug-dota/rotated_rtmdet_l-100e-aug-dota_20221124_224135.json) |

**Note**:

1. We follow the latest metrics from the DOTA evaluation server, original voc format mAP is now mAP50.
2. `IN` means ImageNet pretrain, `COCO` means COCO pretrain.
3. Different from the report, the inference speed here is measured on an NVIDIA 2080Ti GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and with NMS.
