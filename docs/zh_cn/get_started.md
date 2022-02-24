## 模型测试

- 单 GPU 测试
- 单节点多 GPU 测试
- 多节点测试

可以使用以下命令来进行数据集推理。

```shell
# 单 GPU 测试
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# 多 GPU 测试
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]

# slurm 环境下的多节点测试
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments] --launcher slurm
```


示例：

使用RotatedRetinaNet模型在DOTA-1.0数据集上进行推理，并生成用于[官方评测](https://captain-whu.github.io/DOTA/evaluation.html)的压缩文件。（需要先修改 [数据集配置文件](../../configs/_base_/datasets/dotav1.py)）
```shell
python ./tools/test.py  \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/SOME_CHECKPOINT.pth --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```
或者
```shell
./tools/dist_test.sh  \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/SOME_CHECKPOINT.pth 1 --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```

可以将 [数据集配置文件](.../configs/_base_/datasets/dotav1.py) 中的测试集目录改为测试集或者训练测试集目录，用于离线评估。
```shell
python ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/SOME_CHECKPOINT.pth --eval mAP
```
或者
```shell
./tools/dist_test.sh  \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/SOME_CHECKPOINT.pth 1 --eval mAP
```

也可以使用下面的命令将结果可视化。
```shell
python ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/SOME_CHECKPOINT.pth \
  --show-dir work_dirs/vis
```



## 模型训练

### 单 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果想在命令中指定工作目录，可以添加一个参数 `--work_dir ${YOUR_WORK_DIR}`。

### 多 GPU 训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数 [optional arguments] 如下：

- `--no-validate` （**不建议**）：默认情况下，在训练期间会进行测试。使用 `--no-validate`则会在训练期间关闭测试。
- `--work-dir ${WORK_DIR}`：覆盖配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`：从之前的 checkpoint 文件继续训练。

`resume-from` 和 `load-from`的区别：
`resume-from` 同时加载模型权重和优化器状态，也会继承指定 checkpoint 的迭代轮数，经常被用于恢复意外中断的训练。
`load-from` 则是只加载模型权重，它的训练是从头开始的，经常被用于微调模型。

### 多机多 GPU 训练

如果在一个用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMRotate，可以使用脚本 `slurm_train.sh` 进行训练（这个脚本也支持单机训练）。

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

如果只是用 Ethernet 连接了多台机器，可以参考 PyTorch [启动工具](https://pytorch.org/docs/stable/distributed.html#launch-utility)。

通常情况下，如果没有像 InfiniBand 这样的高速网络，训练则会比较慢。

### 在一台机器上启动多个任务

如果你想在一台机器上启动多个任务的话，比如在一个有 8 块 GPU 的机器上启动 2 个需要 4 块GPU的任务，你需要给不同的训练任务指定不同的端口（默认为 29500）来避免冲突。

如果你使用 `dist_train.sh` 来启动训练任务，你可以使用命令来设置端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果你用 Slurm 启动训练任务，则需要修改配置文件（通常是配置文件中从底部开始的第6行）来设置不同的通信端口。

在 `config1.py` 中，设置：

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中，设置：

```python
dist_params = dict(backend='nccl', port=29501)
```

然后你可以使用 `config1.py` 和 `config2.py` 来启动两个任务了。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```
