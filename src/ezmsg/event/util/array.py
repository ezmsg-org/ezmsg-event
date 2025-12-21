import sparse


def sliding_win_oneaxis(s: sparse.SparseArray, nwin: int, axis: int, step: int = 1) -> sparse.SparseArray:
    if -s.ndim <= axis < 0:
        axis = s.ndim + axis
    targ_slices = [slice(_, _ + nwin) for _ in range(0, s.shape[axis] - nwin + 1, step)]
    s = s.reshape(s.shape[:axis] + (1,) + s.shape[axis:])
    full_slices = (slice(None),) * s.ndim
    full_slices = [full_slices[: axis + 1] + (sl,) + full_slices[axis + 2 :] for sl in targ_slices]
    result = sparse.concatenate([s[_] for _ in full_slices], axis=axis)
    # TODO: Profile this approach vs working with coords and values only.
    return result
