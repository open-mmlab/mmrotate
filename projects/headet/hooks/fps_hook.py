import numpy as np

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.device import get_max_cuda_memory, is_cuda_available

import pymongo
import os


@HOOKS.register_module()
class FPSHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """

    def __init__(self, mongo=False):


        self.mongo = mongo
        if self.mongo:
            MONGO = os.getenv('MONGO')
            client = pymongo.MongoClient(MONGO)
            db = client.openmmlab
            self.collection = getattr(db, self.mongo)


    def after_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """


        data = dict()
        data1 = dict()

        data['version'] = self.mongo
        data['experiment_name'] = runner.experiment_name
        data['model_group'] = runner.log_dir.split('/')[-3]
        data['config_file'] = runner.log_dir.split('/')[-2]+'.py'
        data['model_name'] = runner.model_name
        data['max_epochs'] = runner.max_epochs
        data['max_iters'] = runner.max_iters

        for key, value in runner.message_hub.log_scalars.items():
            data1[key] =  np.average(value.data[0])
        data['FPS'] = 1.0/data1['test/time']
        if is_cuda_available():
            data['max_memory'] = self._get_max_memory(runner)

        data.update(data1)

        data['log_dir'] = runner.log_dir
        data['cfg'] = runner.cfg.text
        print(data)

        if self.mongo:
            r = self.collection.insert_one(data)
            print(r)

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.

        Returns:
            The maximum GPU memory occupied by tensors in megabytes for a given
            device.
        """

        device = getattr(runner.model, 'output_device', None)
        return get_max_cuda_memory(device)
