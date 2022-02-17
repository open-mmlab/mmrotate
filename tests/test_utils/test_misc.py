# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

from mmrotate.utils import find_latest_checkpoint


def test_find_latest_checkpoint():
    """Test find latest checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir
        latest = find_latest_checkpoint(path)
        # There are no checkpoints in the path.
        assert latest is None

        path = tmpdir + '/none'
        latest = find_latest_checkpoint(path)
        # The path does not exist.
        assert latest is None
