# Changelog of v1.x

## v1.0.0rc1 (30/12/2022)

### Highlights

- Support [RTMDet](https://arxiv.org/abs/2212.07784) rotated object detection models. The technical report of RTMDet is on [arxiv](https://arxiv.org/abs/2212.07784) (#662)
- Support H2RBox. (#644)

### New Features

- Support PSC (#617)
- Add [`projects/`](./projects/) folder and give an [example](./projects/example_project/README.md) for communities to contribute their projects. (#627)
- Support DIOR Dataset. (#639)

### Bug Fixes

- Fix `get_flops.py` in 1.x. (#646)
- Fix Windows CI. (#621)
- Fix error in rbbox_overlaps. (#620)

### Improvements

- Deprecating old type alias due to new version of numpy (#674)
- Use iof in RRandomCrop. (#660)
- Modify keys in dataset.metainfo to lower case. (#654)
- Add torch 1.13 in CI. (#661)
- Add dockerfile in 1.x. (#631)
- Use mmengine in torchserve deployment. (#616)
- Add `.pre-commit-config-zh-cn.yaml`. (#630)

### New Contributors

- @yuyi1005 made their first contribution in <https://github.com/open-mmlab/mmrotate/pull/617>
- @yxzhao2022 made their first contribution in <https://github.com/open-mmlab/mmrotate/pull/639>
- @YanxingLiu made their first contribution in <https://github.com/open-mmlab/mmrotate/pull/631>

### Contributors

A total of 11 developers contributed to this release.

Thanks @yxzhao2022 @yuyi1005 @YanxingLiu @nijkah @RangeKing @austinmw @liuyanyi @yangxue0827 @zytx121 @RangiLyu @ZwwWayne

## v1.0.0rc0 (7/11/2022)

We are excited to announce the release of MMRotate 1.0.0rc0.
MMRotate 1.0.0rc0 is the first version of MMRotate 1.x, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine), MMRotate 1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.

### Highlights

1. **New engines**. MMRotate 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMRotate 1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **New BoxType design**. We support data structures RotatedBoxes and QuadriBoxes to encapsulate different kinds of bounding boxes. We are migrating to use data structures of boxes to replace the use of pure tensor boxes. This will unify the usages of different kinds of bounding boxes in MMDetection 3.x and MMRotate 1.x to simplify the implementation and reduce redundant codes.

4. **Stronger visualization**. We provide a series of useful tools which are mostly based on brand-new visualizers. As a result, it is more convenient for the users to explore the models and datasets now.

### Breaking Changes

We briefly list the major breaking changes here.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.

#### Dependencies

- MMRotate 1.x relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models in OpenMMLab 2.0 models. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.
- MMRotate 1.x relies on MMCV>=2.0.0rc2. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMRotate 1.x relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provide pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated.
- MMRotate 1.x relies on MMDetection>=3.0.0rc2.

#### Training and testing

- MMRotate 1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMRotate 1.x no longer maintains the building logics of those modules in `mmrotate.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum scheduling has been migrated from `Hook` to `Parameter Scheduler` in MMEngine. Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures to ease the understanding of the components in runner. Users can refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects.

#### Dataset

The Dataset classes implemented in MMRotate 1.x all inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html).

- All the datasets support to serialize the data list to reduce the memory when multiple workers are built to accelerate data loading.

#### Data Transforms

The data transforms in MMRotate 1.x all inherits from those in MMCV>=2.0.0rc2, which follows a new convention in OpenMMLab 2.0 projects.
The changes are listed as below:

- The interfaces are also changed. Please refer to the [API Reference](https://mmrotate.readthedocs.io/en/dev-1.x/)
- The functionality of some data transforms (e.g., `Rotate`) are decomposed into several transforms.

#### Model

The models in MMRotate 1.x all inherits from `BaseModel` in MMEngine, which defines a new convention of models in OpenMMLab 2.0 projects. Users can refer to the [tutorial of model](https://mmengine.readthedocs.io/en/latest/tutorials/model.html) in MMengine for more details. Accordingly, there are several changes as the following:

- The model interfaces, including the input and output formats, are significantly simplified and unified following the new convention in MMRotate 1.x. Specifically, all the input data in training and testing are packed into `inputs` and `data_samples`, where `inputs` contains model inputs like a list of image tensors, and `data_samples` contains other information of the current data sample such as ground truths and model predictions. In this way, different tasks in MMRotate 1.x can share the same input arguments, which makes the models more general and suitable for multi-task learning.
- The model has a data preprocessor module, which is used to pre-process the input data of model. In MMRotate 1.x, the data preprocessor usually does necessary steps to form the input images into a batch, such as padding. It can also serve as a place for some special data augmentations or more efficient data transformations like normalization.
- The internal logic of model have been changed. In MMRotate 0.x, model used `forward_train` and `simple_test` to deal with different model forward logics. In MMRotate 1.x and OpenMMLab 2.0, the forward function has three modes: `loss`, `predict`, and `tensor` for training, inference, and tracing or other purposes, respectively. The forward function calls `self.loss()`, `self.predict()`, and `self._forward()` given the modes `loss`, `predict`, and `tensor`, respectively.

#### Evaluation

MMRotate 1.x mainly implements corresponding metrics for each task, which are manipulated by [Evaluator](https://mmengine.readthedocs.io/en/latest/design/evaluator.html) to complete the evaluation.
In addition, users can build evaluator in MMRotate 1.x to conduct offline evaluation, i.e., evaluate predictions that may not produced by MMRotate, prediction follows our dataset conventions. More details can be find in the [Evaluation Tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html) in MMEngine.

#### Visualization

The functions of visualization in MMRotate 1.x are removed. Instead, in OpenMMLab 2.0 projects, we use [Visualizer](https://mmengine.readthedocs.io/en/latest/design/visualization.html) to visualize data. MMRotate 1.x implements `RotLocalVisualizer` to allow visualization of ground truths, model predictions, and feature maps, etc., at any place. It also supports to dump the visualization data to any external visualization backends such as Tensorboard and Wandb.

### Improvements

- Support quadrilateral box detection (#520)
- Support RotatedCocoMetric (#557)
- Support COCO style annotations (#582)
- Support two new SAR datasets: RSDD and SRSDD (#591)

### Ongoing changes

1. Test-time augmentation: is not implemented yet in this version due to limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools/` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMRotate 1.x.

### Contributors

A total of 8 developers contributed to this release.
Thanks  @DonggeunYu @k-papadakis @liuyanyi  @yangxue0827 @jbwang1997 @zytx121  @RangiLyu @ZwwWayne
