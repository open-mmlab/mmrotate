# 训练 & 测试 (待更新)

## 测试一个模型

- 单个 GPU
- 单个节点多个 GPU
- 多个节点多个 GPU

您可以使用以下命令来推理数据集。

```shell
# 单个 GPU
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# 多个 GPU
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]

# slurm 环境中多个节点
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments] --launcher slurm
```

例子:

在 DOTA-1.0 数据集推理 RotatedRetinaNet 并生成压缩文件用于在线[提交](https://captain-whu.github.io/DOTA/evaluation.html) (首先请修改 [data_root](https://github.com/open-mmlab/mmrotate/tree/main/configs/_base_/datasets/dotav1.py))。

```shell
python ./tools/test.py  \
  configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py \
  checkpoints/SOME_CHECKPOINT.pth --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```

或者

```shell
./tools/dist_test.sh  \
  configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py \
  checkpoints/SOME_CHECKPOINT.pth 1 --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```

您可以修改 [data_root](https://github.com/open-mmlab/mmrotate/tree/main/configs/_base_/datasets/dotav1.py) 中测试集的路径为验证集或训练集路径用于离线的验证。

```shell
python ./tools/test.py \
  configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py \
  checkpoints/SOME_CHECKPOINT.pth --eval mAP
```

或者

```shell
./tools/dist_test.sh  \
  configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py \
  checkpoints/SOME_CHECKPOINT.pth 1 --eval mAP
```

您也可以可视化结果。

```shell
python ./tools/test.py \
  configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py \
  checkpoints/SOME_CHECKPOINT.pth \
  --show-dir work_dirs/vis
```

## 训练一个模型

### 单 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果您想在命令行中指定工作路径，您可以增加参数 `--work_dir ${YOUR_WORK_DIR}`。

### 多 GPU 训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数包括:

- `--no-validate` (**不建议**): 默认情况下代码将在训练期间进行评估。通过设置 `--no-validate` 关闭训练期间进行评估。
- `--work-dir ${WORK_DIR}`: 覆盖配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`: 从以前的检查点恢复训练。

`resume-from` 和 `load-from` 的不同点：

`resume-from` 读取模型的权重和优化器的状态，并且 epoch 也会继承于指定的检查点。通常用于恢复意外中断的训练过程。
`load-from` 只读取模型的权重并且训练的 epoch 会从 0 开始。通常用于微调。

### 使用多台机器训练

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

### 使用 Slurm 来管理任务

如果您在 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMRotate，您可以使用脚本 `slurm_train.sh` (此脚本还支持单机训练)。

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

如果您有多台机器联网，您可以参考 PyTorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility)。
如果您没有像 InfiniBand 这样的高速网络，训练速度通常会很慢。

### 在一台机器上启动多个作业

如果您在一台机器上启动多个作业，如在一台机器上使用 8 张 GPU 训练 2 个作业，每个作业使用 4 张 GPU ，您需要为每个作业指定不同的端口号(默认为 29500 )进而避免通讯冲突。

如果您使用 `dist_train.sh` 启动训练，您可以在命令行中指定端口号。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果您通过 Slurm 启动训练，您需要修改配置文件(通常是配置文件底部的第 6 行)进而设置不同的通讯端口。

在 `config1.py` 中,

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中,

```python
dist_params = dict(backend='nccl', port=29501)
```

之后您可以使用 `config1.py` 和 `config2.py` 开启两个作业。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```
