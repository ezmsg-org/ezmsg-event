import math

import numpy as np
import pytest
import sparse

from ezmsg.event.util.array import sliding_win_oneaxis


@pytest.mark.parametrize("nwin", [20, 10])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("step", [10, 1, 20])
def test_sliding_win_oneaxis(nwin: int, axis: int, step: int):
    rng = np.random.default_rng()
    s = sparse.random((100, 50, 30), density=0.1, random_state=rng)
    result = sliding_win_oneaxis(s, nwin, axis, step)

    n_steps = int(math.ceil((s.shape[axis] - nwin + 1) / step))
    exp_shape = s.shape[:axis] + (n_steps, nwin) + s.shape[axis + 1 :]
    assert result.shape == exp_shape
