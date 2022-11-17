# Dummy ResNet Wrapper

This is an example README for community `projects/`. We have provided detailed explanations for each field in the form of html comments, which are visible when you read the source of this README file. If you wish to submit your project to our main repository, then all the fields in this README are mandatory for others to understand what you have achieved in this implementation. For more details, read our [contribution guide](https://mmrotate.readthedocs.io/en/1.x/notes/contribution_guide.html) or approach us in [Discussions](https://github.com/open-mmlab/mmrotate/discussions).

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

This project implements a dummy ResNet wrapper, which literally does nothing new but prints "hello world" during initialization.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMRotate's root directory, run the following command to train the model:

```bash
python tools/train.py projects/example_project/configs/r3det-oc_dummy-resnet_fpn_1x_dota.py
```

### Testing commands

In MMRotate's root directory, run the following command to test the model:

```bash
python tools/test.py projects/example_project/confsigs/r3det-oc_dummy-resnet_fpn_1x_dota.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/r3det/README.md#results-and-models)
You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                      Configs                                       |         Download         |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :--------------------------------------------------------------------------------: | :----------------------: |
| ResNet50 (1024,1024,200) | 69.80 |  oc   |   1x    |   3.54   |      12.4      |  -  |     2      | [r3det-oc_dummy-resnet_fpn_1x_dota](confsigs/r3det-oc_dummy-resnet_fpn_1x_dota.py) | [model](<>) \| [log](<>) |

## Citation

<!-- You may remove this section if not applicable. -->

```bibtex
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {7331–7334},
  numpages = {4},
  year={2022}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmrotate.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [ ] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmrotate/blob/766185ed317f99379cb14035a6f9e5cf8a5340ad/mmrotate/structures/bbox/box_converters.py#L61-L78) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmrotate/blob/766185ed317f99379cb14035a6f9e5cf8a5340ad/tests/test_structures/test_bbox/test_box_converters.py#L43-L52) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/r3det/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/r3det/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
