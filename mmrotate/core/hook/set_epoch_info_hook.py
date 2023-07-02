# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHookwithMaxEpoch(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        cur_epoch = runner.epoch
        max_epochs = runner.max_epochs
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(cur_epoch, max_epochs)
