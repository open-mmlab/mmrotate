# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmdetection/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## MMCV Installation

- Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

Compatible MMCV, MMDetection and MMRotate versions are shown as below. Please install the correct version of them to avoid installation issues.

| MMRotate version |       MMCV version        | MMDetection version |
| :--------------: | :-----------------------: | :-----------------: |
|       main       | mmcv-full>=1.5.3, \<1.8.0 |   mmdet >= 2.25.1   |
|      0.3.3       | mmcv-full>=1.5.3, \<1.7.0 |   mmdet >= 2.25.1   |
|      0.3.2       | mmcv-full>=1.5.3, \<1.7.0 |   mmdet >= 2.25.1   |
|      0.3.1       | mmcv-full>=1.4.5, \<1.6.0 |   mmdet >= 2.22.0   |
|      0.3.0       | mmcv-full>=1.4.5, \<1.6.0 |   mmdet >= 2.22.0   |
|      0.2.0       | mmcv-full>=1.4.5, \<1.5.0 |   mmdet >= 2.19.0   |
|      0.1.1       | mmcv-full>=1.4.5, \<1.5.0 |   mmdet >= 2.19.0   |
|      0.1.0       | mmcv-full>=1.4.5, \<1.5.0 |   mmdet >= 2.19.0   |

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'".

  1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
  2. Install mmcv-full following the [installation instruction](install#best-practices).

## PyTorch/CUDA Environment

- "invalid device function" or "no kernel image is available for execution".

  1. Check if your cuda runtime version (under `/usr/local/`), `nvcc --version` and `conda list cudatoolkit` version match.
  2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture.
     You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV.
     The GPU arch table could be found [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
     i.e. run `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv-full` to build MMCV for Volta GPUs.
     The compatibility issue could happen when using old GPUS, e.g., Tesla K80 (3.7) on colab.
  3. Check whether the running environment is the same as that when mmcv/mmdet has compiled.
     For example, you may compile mmcv using CUDA 10.0 but run it on CUDA 9.0 environments.

- "undefined symbol" or "cannot open xxx.so".

  1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check whether the CUDA/GCC runtimes are the same as those used for compiling mmcv,
     i.e. run `python mmdet/utils/collect_env.py` to see if `"MMCV Compiler"`/`"MMCV CUDA Compiler"` is the same as `"GCC"`/`"CUDA_HOME"`.
  2. If those symbols are PyTorch symbols (e.g., symbols containing caffe, aten, and TH), check whether the PyTorch version is the same as that used for compiling mmcv.
  3. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built by and running on the same environment.

- "setuptools.sandbox.UnpickleableException: DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")"

  1. If you are using miniconda rather than anaconda, check whether Cython is installed as indicated in [#3379](https://github.com/open-mmlab/mmdetection/issues/3379).
     You need to manually install Cython first and then run command `pip install -r requirements.txt`.
  2. You may also need to check the compatibility between the `setuptools`, `Cython`, and `PyTorch` in your environment.

- "Segmentation fault".

  1. Check you GCC version and use GCC 5.4. This usually caused by the incompatibility between PyTorch and the environment (e.g., GCC \< 4.9 for PyTorch). We also recommend the users to avoid using GCC 5.5 because many feedbacks report that GCC 5.5 will cause "segmentation fault" and simply changing it to GCC 5.4 could solve the problem.

  2. Check whether PyTorch is correctly installed and could use CUDA op, e.g. type the following command in your terminal.

     ```shell
     python -c 'import torch; print(torch.cuda.is_available())'
     ```

     And see whether they could correctly output results.

  3. If Pytorch is correctly installed, check whether MMCV is correctly installed.

     ```shell
     python -c 'import mmcv; import mmcv.ops'
     ```

     If MMCV is correctly installed, then there will be no issue of the above two commands.

  4. If MMCV and Pytorch is correctly installed, you man use `ipdb`, `pdb` to set breakpoints or directly add 'print' in mmdetection code and see which part leads the segmentation fault.

## E2CNN

- "ImportError: cannot import name 'container_bacs' from 'torch.\_six'"

  1. This is because `container_abcs` has been removed since PyTorch 1.9.

  2. Replace

     ```shell
     from torch.six import container_abcs
     ```

     in `python3.7/site-packages/e2cnn/nn/modules/module_list.py` with

     ```shell
     TORCH_MAJOR = int(torch.__version__.split('.')[0])
     TORCH_MINOR = int(torch.__version__.split('.')[1])
     if TORCH_MAJOR ==1 and TORCH_MINOR < 8:
         from torch.six import container_abcs
     else:
         import collections.abs as container_abcs
     ```

  3. Or downgrade the version of Pytorch.

## Training

- "Loss goes Nan"

  1. Check if the dataset annotations are valid: zero-size bounding boxes will cause the regression loss to be Nan due to the commonly used transformation for box regression. Some small size (width or height are smaller than 1) boxes will also cause this problem after data augmentation (e.g., instaboost). So check the data and try to filter out those zero-size boxes and skip some risky augmentations on the small-size boxes when you face the problem.
  2. Reduce the learning rate: the learning rate might be too large due to some reasons, e.g., change of batch size. You can rescale them to the value that could stably train the model.
  3. Extend the warmup iterations: some models are sensitive to the learning rate at the start of the training. You can extend the warmup iterations, e.g., change the `warmup_iters` from 500 to 1000 or 2000.
  4. Add gradient clipping: some models requires gradient clipping to stabilize the training process. The default of `grad_clip` is `None`, you can add gradient clippint to avoid gradients that are too large, i.e., set `optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))` in your config file. If your config does not inherits from any basic config that contains `optimizer_config=dict(grad_clip=None)`, you can simply add `optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))`.

- "GPU out of memory"

  1. There are some scenarios when there are large amounts of ground truth boxes, which may cause OOM during target assignment. You can set `gpu_assign_thr=N` in the config of assigner thus the assigner will calculate box overlaps through CPU when there are more than N GT boxes.
  2. Set `with_cp=True` in the backbone. This uses the sublinear strategy in PyTorch to reduce GPU memory cost in the backbone.
  3. Try mixed precision training by setting `fp16 = dict(loss_scale='dynamic')` in the config file.

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

  1. This error indicates that your module has parameters that were not used in producing loss. This phenomenon may be caused by running different branches in your code in DDP mode.
  2. You can set `find_unused_parameters = True` in the config to solve the above problems or find those unused parameters manually.

## Evaluation

- COCO Dataset, AP or AR = -1
  1. According to the definition of COCO dataset, the small and medium areas in an image are less than 1024 (32\*32), 9216 (96\*96), respectively.
  2. If the corresponding area has no object, the result of AP and AR will set to -1.
