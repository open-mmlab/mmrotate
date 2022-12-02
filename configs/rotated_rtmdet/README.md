# RTMDet-R

<!-- [ALGORITHM] -->

## Abstract

Our tech-report will be released soon.

<div align=center>
<img src="https://user-images.githubusercontent.com/11705038/204995787-ef739910-e196-42c7-a9db-c9c8e28a494d.jpg" height="360"/>
</div>

## Results and Models

### DOTA-v1.0

|  Backbone   | size | pretrain |  Aug  | mmAP  | mAP50 | mAP75 | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                          Config                          |                                                           Download                                                           |
| :---------: | :--: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny | 1024 |   IMP    |  RR   | 47.37 | 75.36 | 50.64 |   4.88    |  20.45   |         4.40         |        [config](./rotated_rtmdet_tiny-3x-dota.py)        |        [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/) \| [log](<>)        |
| RTMDet-tiny | 1024 |   IMP    | MS+RR | 53.59 | 79.82 | 58.87 |   4.88    |  20.45   |         4.40         |      [config](./rotated_rtmdet_tiny-3x-dota_ms.py)       |      [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota_ms/) \| [log](<>)       |
|  RTMDet-s   | 1024 |   IMP    |  RR   | 48.16 | 76.93 | 50.59 |   8.86    |  37.62   |         4.86         |         [config](./rotated_rtmdet_s-3x-dota.py)          |         [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/) \| [log](<>)          |
|  RTMDet-s   | 1024 |   IMP    | MS+RR | 54.43 | 79.98 | 60.07 |   8.86    |  37.62   |         4.86         |        [config](./rotated_rtmdet_s-3x-dota_ms.py)        |        [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota_ms/) \| [log](<>)        |
|  RTMDet-m   | 1024 |   IMP    |  RR   | 50.56 | 78.24 | 54.47 |   24.67   |  99.76   |         7.82         |         [config](./rotated_rtmdet_m-3x-dota.py)          |         [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/) \| [log](<>)          |
|  RTMDet-m   | 1024 |   IMP    | MS+RR | 55.00 | 80.26 | 61.26 |   24.67   |  99.76   |         7.82         |        [config](./rotated_rtmdet_m-3x-dota_ms.py)        |        [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota_ms/) \| [log](<>)        |
|  RTMDet-l   | 1024 |   IMP    |  RR   | 51.01 | 78.85 | 55.21 |   52.27   |  204.21  |        10.82         |         [config](./rotated_rtmdet_l-3x-dota.py)          |         [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota/) \| [log](<>)          |
|  RTMDet-l   | 1024 |   IMP    | MS+RR | 55.52 | 80.54 | 61.47 |   52.27   |  204.21  |        10.82         |        [config](./rotated_rtmdet_l-3x-dota_ms.py)        |        [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms/) \| [log](<>)        |
|  RTMDet-l   | 1024 |   COP    | MS+RR | 56.74 | 81.33 | 63.45 |   52.27   |  204.21  |        10.82         | [config](./rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-dota_ms/) \| [log](<>) |

### HRSC

|  Backbone   | size | pretrain | Aug | mAP 07 | mAP 12 | Params(M) | FLOPS(G) |                   Config                   |                                                                                                                                           Download                                                                                                                                           |
| :---------: | :--: | :------: | :-: | :----: | :----: | :-------: | :------: | :----------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny | 800  |   IMP    | RR  |  90.6  |  97.1  |   4.88    |  12.54   | [config](./rotated_rtmdet_tiny-9x-hrsc.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-9x-hrsc/rotated_rtmdet_tiny-9x-hrsc-9f2e3ca6.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-9x-hrsc/rotated_rtmdet_tiny-9x-hrsc-20221125_145920.json) |

**Note**:

1. By default, DOTA-v1.0 dataset trained with 3x schedule and HRSC dataset trained with 9x schedule.
2. We follow the latest metrics from the DOTA evaluation server, original voc format mAP is now mAP50.
3. `IMP` means ImageNet pretrain, `COP` means COCO pretrain.
4. The inference speed is measured on an NVIDIA 2080Ti GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and
   without NMS.
5. We also provide config with mixup and mosaic for longer schedule.
