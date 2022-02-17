# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def _check_fields(results, pipeline_results, keys):
    """Check data in fields from two results are same."""
    for key in keys:
        assert np.equal(results[key], pipeline_results[key]).all()
        assert results[key].dtype == pipeline_results[key].dtype
