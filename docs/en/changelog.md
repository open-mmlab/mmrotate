## Changelog

### v0.3.3 (27/10/2022)

#### Bug Fixes

- Fix reppoint bug fix when negative image training (#396)
- Fix bug in oriented_reppoints_head.py (#424)
- Fix mmcv-full version (#423)

#### Improvements

- Update issue templates to main branch (#579)
- Fix lint of dev branch (#578)

#### Documentations

- Update citation (#425)
- Fix markdown version when building docs (#414)

#### Contributors

A total of 5 developers contributed to this release.
Thanks @yangxue0827, @ZwwWayne, @MinkiSong, @zytx121, @RangiLyu

### v0.3.2 (6/7/2022)

#### Highlight

- Support Oriented Reppoints (CVPR'22) (#286)
- Support ConvNeXt backbone (CVPR'22) (#343)

#### New Features

- Support RMosaic. (#344)

#### Bug Fixes

- Fix max_coordinate in multiclass_nms_rotated. (#346)
- Fix bug in PolyRandomRotate. (#366)
- Fix memory shortage when using huge_image_demo.py. (#368)

#### Improvements

- Update README.md and INSTALL.md. (#342)
- Fix typo in rotated_fcos_head. (#354)
- Update checkpoint and eval interval of base config. (#347)
- Fix mdformat version to support python3.6 & Add mim to extras_require in setup.py. (#359)
- Add mim test in CI. (#374)

#### Contributors

A total of 9 developers contributed to this release.
Thanks @LiWentomng @heiyuxiaokai @JinYuannn @sltlls @liuyanyi  @yangxue0827 @jbwang1997 @zytx121 @ZwwWayne

### v0.3.1 (6/6/2022)

#### Highlight

- Support Rotated FCOS (#223)

#### New Features

- Update PolyRandomRotate to support discrete angle value. (#281)
- Support RRandomCrop. (#322)
- Support mask in merge_results and huge_image_demo.py. (#280)
- Support don't filter images without ground truths. (#323)
- Add MultiImageMixDataset in build_dataset. (#331)

#### Bug Fixes

- Fix error in Windows CI. (#324)
- Fix data path error in config files. (#328)
- Fix bug when visualize the HRSC2016 detect results. (#329)

#### Improvements

- Add torchserve doc in zh_cn. (#287)
- Fix doc typo in README. (#284)
- Configure Myst-parser to parse anchor tag (#305 #308)
- Replace markdownlint with mdformat for avoiding installing ruby. (#306)
- Fix typo about split gap of multi scale. (#272)

#### Contributors

A total of 7 developers contributed to this release.
Thanks @liuyanyi @nijkah @remi-or @yangxue0827 @jbwang1997 @zytx121 @ZwwWayne

### v0.3.0 (29/4/2022)

#### Highlight

- Support TorchServe (#160)
- Support Rotated ATSS (CVPR'20) (#179)

#### New Features

- Update performance of ReDet on HRSC2016. (#203)

- Upgrage visualization to custom colors of different classes. This requires mmdet>=2.22.0. (#187, #267, #270)

- Update Stable KLD, which solve the Nan issue of KLD training. (#183)

- Support setting dataloader arguments in config and add functions to handle config compatibility. (#215)
  The comparison between the old and new usages is as below.

  <table align="center">
    <thead>
        <tr align='center'>
            <td>Before v0.2.0</td>
            <td>Since v0.3.0 </td>
        </tr>
    </thead>
    <tbody><tr valign='top'>
    <th>

  ```python
  data = dict(
      samples_per_gpu=2, workers_per_gpu=2,
      train=dict(type='xxx', ...),
      val=dict(type='xxx', samples_per_gpu=4, ...),
      test=dict(type='xxx', ...),
  )
  ```

  </th>
    <th>

  ```python
  # A recommended config that is clear
  data = dict(
      train=dict(type='xxx', ...),
      val=dict(type='xxx', ...),
      test=dict(type='xxx', ...),
      # Use different batch size during inference.
      train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=2),
      val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
      test_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
  )

  # Old style still works but allows to set more arguments about data loaders
  data = dict(
      samples_per_gpu=2,  # only works for train_dataloader
      workers_per_gpu=2,  # only works for train_dataloader
      train=dict(type='xxx', ...),
      val=dict(type='xxx', ...),
      test=dict(type='xxx', ...),
      # Use different batch size during inference.
      val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
      test_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
  )
  ```

  </th></tr>
  </tbody></table>

- Add [get_flops](tools/analysis_tools/get_flops.py) tool (#176)

#### Bug Fixes

- Fix bug about rotated anchor inside flags. (#197)
- Fix Nan issue of GWD. (#206)
- Fix bug in eval_rbbox_map when labels_ignore is None. (#209)
- Fix bug of 'RoIAlignRotated' object has no attribute 'output_size' (#213)
- Fix bug in unit test for datasets. (#222)
- Fix bug in rotated_reppoints_head. (#246)
- Fix GPG key error in CI and docker. (#269)

#### Improvements

- Update citation of mmrotate in README.md (#263)
- Update the introduction of SASM (AAAI'22) (#184)
- Fix doc typo in Config File and Model Zoo. (#199)
- Unified RBox definition in doc. (#234)

#### Contributors

A total of 7 developers contributed to this release.
Thanks @nijkah @GamblerZSY @liuyanyi @yangxue0827 @jbwang1997 @zytx121 @ZwwWayne

### v0.2.0 (30/3/2022)

#### New Features

- Support Circular Smooth Label (CSL, ECCV'20) (#153)
- Support multiple machines dist_train (#143)
- Add [browse_dataset](tools/misc/browse_dataset.py) tool (#98)
- Add [gather_models](.dev_scripts/gather_models.py) script (#162)

#### Bug Fixes

- Remove in-place operations in rbbox_overlaps (#155)
- Fix bug in docstring. (#137)
- Fix bug in HRSCDataset with `clasesswise=ture` (#175)

#### Improvements

- Add Chinese translation of `docs/zh_cn/tutorials/customize_dataset.md` (#65)
- Add different seeds to different ranks (#102)
- Update from-scratch install script in install.md (#166)
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
