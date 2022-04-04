Apart from training/testing scripts, We provide lots of useful tools under the
 `tools/` directory.

## Log Analysis

`tools/analysis_tools/analyze_logs.py` plots loss/mAP curves given a training
 log file. Run `pip install seaborn` first to install the dependency.

 ```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
 ```

![loss curve image](https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/loss_curve.png)

Examples:

- Plot the classification loss of some run.

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
    ```

- Compare the bbox mAP of two runs in the same figure.

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
    ```

- Compute the average training speed.

    ```shell
    python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
    ```

    The output is expected to be like the following.

    ```text
    -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
    slowest epoch 11, average time is 1.2024
    fastest epoch 1, average time is 1.1909
    time std over epochs is 0.0028
    average iter time: 1.1959 s/iter
    ```

## Visualization

### Visualize Datasets

`tools/misc/browse_dataset.py` helps the user to browse a detection dataset (both
 images and bounding box annotations) visually, or save the image to a
  designated directory.

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

## Model Complexity

`tools/analysis_tools/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the results like this.

```text
==============================
Input shape: (3, 1024, 1024)
Flops: 215.92 GFLOPs
Params: 36.42 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the
 number is absolutely correct. You may well use the result for simple
  comparisons, but double check it before you adopt it in technical reports or papers.

1. FLOPs are related to the input shape while parameters are not. The default
 input shape is (1, 3, 1024, 1024).
2. Some operators are not counted into FLOPs like DCN and custom operators. So models with dcn such as S<sup>2</sup>A-Net and RepPoints based model got wrong flops.
Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.
3. The FLOPs of two-stage detectors is dependent on the number of proposals.

### Prepare a model for publishing

`tools/model_converters/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append the hash id to the
 filename.

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/model_converters/publish_model.py work_dirs/rotated_faster_rcnn/latest.pth rotated_faster_rcnn_r50_fpn_1x_dota_le90_20190801.pth
```

The final output filename will be `rotated_faster_rcnn_r50_fpn_1x_dota_le90_20190801-{hash id}.pth`.

## Benchmark

### FPS Benchmark

`tools/analysis_tools/benchmark.py` helps users to calculate FPS. The FPS value includes model forward and post-processing. In order to get a more accurate value, currently only supports single GPU distributed startup mode.

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT} tools/analysis_tools/benchmark.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    [--repeat-num ${REPEAT_NUM}] \
    [--max-iter ${MAX_ITER}] \
    [--log-interval ${LOG_INTERVAL}] \
    --launcher pytorch
```

Examples: Assuming that you have already downloaded the `Rotated Faster R-CNN` model checkpoint to the directory `checkpoints/`.

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py \
       checkpoints/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth \
       --launcher pytorch
```

## Miscellaneous

### Print the entire config

`tools/misc/print_config.py` prints the whole config verbatim, expanding all its
 imports.

```shell
python tools/misc/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

## Confusion Matrix

A confusion matrix is a summary of prediction results.

`tools/analysis_tools/confusion_matrix.py` can analyze the prediction results and plot a confusion matrix table.

First, run `tools/test.py` to save the `.pkl` detection results.

Then, run

```
python tools/analysis_tools/confusion_matrix.py ${CONFIG}  ${DETECTION_RESULTS}  ${SAVE_DIR} --show
```

And you will get a confusion matrix like this:

![confusion_matrix_example](https://raw.githubusercontent.com/liuyanyi/doc-image/main/confusion_matrix.png)
