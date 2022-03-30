## Changelog

### v0.2.0 (30/3/2022)

#### New Features

- Support Circular Smooth Label (CSL, ECCV'20) (#153)
- Support multiple machines dist_train (#143)
- Add [browse_dataset](tools/misc/browse_dataset.py) tool (#98)
- Add [gather_models](.dev_scripts/gather_models.py) script (#162)

#### Bug Fixes

- Remove in-place operations in rbbox_overlaps (#155)

#### Improvements

- Add Chinese translation of `docs/zh_cn/tutorials/customize_dataset.md` (#65)
- Add different seeds to different ranks (#102)
- Add install command in README(_zh-CN).md (#166)
- Improve the arguments of all mmrotate scripts (#168)

#### Contributors

A total of 6 developers contributed to this release.
Thanks @zytx121 @yangxue0827 @ZwwWayne @jbwang1997 @canoe-Z @matrixgame2018

### v0.1.1 (14/3/2022)

#### New Features

- Support [huge image inference](deom/huge_image_demo.py) (#34)
- Support HRSC Dataset (#96)
- Support mixed precision training (#72)
- Add [colab tutorial](demo/MMRotate_Tutorial.ipynb) for beginners (#66)
- Add inference speed statistics [tool](tools/analysis_tools/benchmark.py) (#86)
- Add confusion matrix analysis [tool](tools/analysis_tools/confusion_matrix.py) (#93)

#### Bug Fixes

- Fix URL error of Swin pretrained model (#111)
- Fix bug for SASM during training (#105)
- Fix rbbox_overlaps abnormal when the box is too small (#61)
- Fix bug for visualization (#12, #81)
- Fix stuck when compute mAP (#14, #52)
- Fix 'RoIAlignRotated' object has no attribute 'out_size' bug (#51)
- Add missing init_cfg in dense head (#37)
- Fix install an additional mmcv (#17)
- Fix typos in docs (#3, #11, #36)

#### Improvements

- Move `eval_rbbox_map` from `mmrotate.datasets` to `mmrotate.core.evaluation` (#73)
- Add  Windows CI (#31)
- Add copyright commit hook (#30)
- Add Chinese translation of `docs/zh_cn/get_started.md` (#16)
- Add Chinese translation of `docs/zh_cn/tutorials/customize_runtime.md` (#22)
- Add Chinese translation of ` docs/zh_cn/tutorials/customize_config.md` (#23)
- Add Chinese translation of `docs/zh_cn/tutorials/customize_models.md` (#27)
- Add Chinese translation of `docs/zh_cn/model_zoo.md` (#28)
- Add Chinese translation of `docs/zh_cn/faq.md` (#33)

#### Contributors

A total of 13 developers contributed to this release.
Thanks @zytx121 @yangxue0827 @jbwang1997 @liuyanyi @DangChuong-DC @RangeKing @liufeinuaa @np-csu @akmalulkhairin @SheffieldCao @BrotherHappy @Abyssaledge  @q3394101
