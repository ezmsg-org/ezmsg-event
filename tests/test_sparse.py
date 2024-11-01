import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.sparse import to_dense


@pytest.mark.parametrize("sparse_input", [True, False])
def test_to_dense(sparse_input: bool):
    arr_shape = (100, 50, 30)
    if sparse_input:
        rng = np.random.default_rng()
        data = sparse.random(arr_shape, density=0.1, random_state=rng)
    else:
        data = np.random.rand(*arr_shape)
    in_msg = AxisArray(data=data, dims=["time", "ch", "freq"], key="test_to_dense")

    proc = to_dense()
    out_msg = proc.send(in_msg)
    assert out_msg.data.shape == in_msg.data.shape
    assert isinstance(out_msg.data, np.ndarray)
    if sparse_input:
        assert np.array_equal(out_msg.data[out_msg.data > 0], in_msg.data.data)
