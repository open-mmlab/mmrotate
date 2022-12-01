# RTMDet-R

<!-- [ALGORITHM] -->

## Abstract

Our tech-report will be released soon.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/192182907-f9a671d6-89cb-4d73-abd8-c2b9dada3c66.png"/>
</div>

## Results and Models

### DOTA-v1.0

|  Backbone   | size | pretrain |  Aug  |  mAP  | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                          Config                          |         Download          |
| :---------: | :--: | :------: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :-----------------------: |
| RTMDet-tiny | 1024 |   IMP    |  RR   | 75.60 |   4.88    |  20.45   |         4.40         |        [config](./rotated_rtmdet_tiny-3x-dota.py)        | [model](<>) \|  [log](<>) |
| RTMDet-tiny | 1024 |   IMP    | MS+RR | 79.82 |   4.88    |  20.45   |         4.46         |      [config](./rotated_rtmdet_tiny-3x-dota_ms.py)       | [model](<>) \|  [log](<>) |
|  RTMDet-s   | 1024 |   IMP    |  RR   | 76.93 |   8.86    |  37.62   |         4.86         |         [config](./rotated_rtmdet_s-3x-dota.py)          | [model](<>) \|  [log](<>) |
|  RTMDet-s   | 1024 |   IMP    | MS+RR | 79.98 |   8.86    |  37.62   |         4.86         |        [config](./rotated_rtmdet_s-3x-dota_ms.py)        | [model](<>) \|  [log](<>) |
|  RTMDet-m   | 1024 |   IMP    |  RR   | 78.24 |   24.67   |  99.76   |         7.82         |         [config](./rotated_rtmdet_m-3x-dota.py)          | [model](<>) \|  [log](<>) |
|  RTMDet-m   | 1024 |   IMP    | MS+RR | 80.26 |   24.67   |  99.76   |         7.82         |        [config](./rotated_rtmdet_m-3x-dota_ms.py)        | [model](<>) \|  [log](<>) |
|  RTMDet-l   | 1024 |   IMP    |  RR   | 78.85 |   52.27   |  204.21  |        10.82         |         [config](./rotated_rtmdet_l-3x-dota.py)          | [model](<>) \|  [log](<>) |
|  RTMDet-l   | 1024 |   IMP    | MS+RR | 80.54 |   52.27   |  204.21  |        10.82         |        [config](./rotated_rtmdet_l-3x-dota_ms.py)        | [model](<>) \|  [log](<>) |
|  RTMDet-l   | 1024 |   COP    | MS+RR | 81.33 |   52.27   |  204.21  |        10.82         | [config](./rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py) | [model](<>) \|  [log](<>) |

### HRSC

|  Backbone   | size | pretrain | Aug | mAP 07 | mAP 12 | Params(M) | FLOPS(G) |                   Config                   |         Download          |
| :---------: | :--: | :------: | :-: | :----: | :----: | :-------: | :------: | :----------------------------------------: | :-----------------------: |
| RTMDet-tiny | 800  |   IMP    | RR  |  90.6  |  97.1  |   4.88    |  12.54   | [config](./rotated_rtmdet_tiny-9x-hrsc.py) | [model](<>) \|  [log](<>) |

**Note**:

1. By default, DOTA-v1.0 dataset trained with 3x schedule and HRSC dataset trained with 9x schedule.
2. `IMP` means ImageNet pretrain, `COP` means COCO pretrain.
3. The inference speed is measured on an NVIDIA 2080Ti GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and
   without NMS.
4. We also provide config with mixup and mosaic for longer schedule.
