## Changelog

### v0.1.1 (18/3/2022)

#### New Features

- Support [huge image inference](deom/huge_image_demo.py) (#34)
- Support mix precision training (#72)
- Add [colab tutorial](demo/MMRotate_Tutorial.ipynb) for beginners (#66)
- Add confusion matrix [analysis tool](tools/analysis_tools/confusion_matrix.py) (#93)

#### Bug Fixes

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
