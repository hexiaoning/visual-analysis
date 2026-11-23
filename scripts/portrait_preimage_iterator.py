"""Enumerate bounded-sum RGB8 block channels in cyclic lexicographic order."""

import operator

import numpy as np


def _least_row(row, required, digit_max):
    """Write the lexicographically least digit string with this exact sum."""
    row.fill(0)
    if not required:
        return
    full, remainder = divmod(required, digit_max)
    if full:
        row[len(row) - full:] = digit_max
    if remainder:
        row[len(row) - full - 1] = remainder


def _advance_row(row, lower, upper, digit_max):
    """Increment the rightmost feasible position and minimize its suffix."""
    prefix = int(row.sum(dtype=np.int64))
    for column in range(len(row) - 1, -1, -1):
        value = int(row[column])
        if value < digit_max and prefix < upper:
            # Increasing this position cannot make the lower bound unreachable:
            # the old suffix already supplied enough, and could be reused.
            row[column] = value + 1
            _least_row(row[column + 1:], max(0, lower - prefix - 1), digit_max)
            return True
        prefix -= value
    return False


def _inputs(lower, upper, block_length, start):
    if isinstance(block_length, (bool, np.bool_)):
        raise ValueError("block_length must be a positive integer")
    try:
        length = operator.index(block_length)
    except TypeError as exc:
        raise ValueError("block_length must be a positive integer") from exc
    if not 0 < length <= np.iinfo(np.int64).max // 255:
        raise ValueError("block_length exceeds the positive int64 sum range")

    low, high = np.asarray(lower), np.asarray(upper)
    if (low.ndim != 1 or high.shape != low.shape
            or low.dtype != np.int64 or high.dtype != np.int64):
        raise ValueError("lower and upper must be matching one-dimensional int64 arrays")
    if np.any(low < 0) or np.any(high > length * 255) or np.any(low > high):
        raise ValueError("each interval must satisfy 0 <= lower <= upper <= 255 * block_length")
    low, high = low.copy(), high.copy()
    shape = (len(low), length)
    first = None
    if start is not None:
        initial = np.asarray(start)
        if initial.shape != shape or initial.dtype != np.uint8:
            raise ValueError("start must be a uint8 matrix of the requested shape")
        first = initial.copy(order="C")
        totals = first.sum(axis=1, dtype=np.int64)
        if np.any(totals < low) or np.any(totals > high):
            raise ValueError("start falls outside the sum bounds")
    return low, high, length, first


def _walk(low, high, length, digit_max, first):
    if first is None:
        current = np.empty((len(low), length), dtype=np.uint8)
        for row, minimum in zip(current, low):
            _least_row(row, int(minimum), digit_max)
    else:
        current = first.copy()
    # A singleton row never carries. This also avoids scanning long constant
    # blocks in a photograph when the variable rows occur earlier in the array.
    fixed = (low == high) & ((low == 0) | (low == digit_max * length) | (length == 1))
    variable = np.flatnonzero(~fixed)[::-1]
    wrapped = False
    while True:
        yield current.copy()
        for index in variable:
            minimum = int(low[index])
            if _advance_row(current[index], minimum, int(high[index]), digit_max):
                break
            _least_row(current[index], minimum, digit_max)
        else:
            if first is None:
                return
            wrapped = True
        if wrapped and np.array_equal(current, first):
            return


def compatible_blocks(lower, upper, block_length, start=None):
    """Yield independent uint8 matrices with inclusive per-row int64 sum bounds.

    The default first matrix is lexicographically least in row-major order.
    A supplied compatible first matrix rotates the same sequence. Exhaustion
    visits every compatible matrix exactly once. No prefix or suffix of the
    sequence is cached. Zero rows have one solution, the empty matrix.

    Invalid bounds, including an empty interval, are reported as ValueError.
    """
    low, high, length, first = _inputs(lower, upper, block_length, start)
    yield from _walk(low, high, length, 255, first)


def _representative(codes, width, low, high, first):
    """Choose compatible values inside these bins, preserving a feasible anchor."""
    floor = (codes.astype(np.uint16) * width).astype(np.uint8)
    if first is None:
        result = floor + (width - 1) // 2
    else:
        result = np.clip(first, floor, floor + (width - 1))
    totals = result.sum(axis=1, dtype=np.int64)
    target = np.clip(totals, low, high)
    change = totals != target
    if np.any(change):
        # All digits have equal bin width. Distribute the required offset
        # from the bin floors evenly, with at most one extra unit per digit.
        rows = floor[change]
        offset = target[change] - rows.sum(axis=1, dtype=np.int64)
        level, remainder = np.divmod(offset, codes.shape[1])
        extra = np.arange(codes.shape[1]) < remainder[:, None]
        result[change] = rows + level[:, None] + extra
    return result


def cover_blocks(lower, upper, block_length, discard_bits=0, start=None):
    """Yield one compatible uint8 matrix in every feasible quantization cell.

    Discarding b low bits groups each digit into bins of width w=2**b. Each
    compatible matrix and its cell's representative differ by at most w-1 in
    every channel value. This covers all compatible matrices on exhaustion.
    A supplied compatible start is the first representative. With zero bits
    discarded the sequence is exactly compatible_blocks(...).
    """
    if isinstance(discard_bits, (bool, np.bool_)):
        raise ValueError("discard_bits must be an integer in [0, 8]")
    try:
        bits = operator.index(discard_bits)
    except TypeError as exc:
        raise ValueError("discard_bits must be an integer in [0, 8]") from exc
    if not 0 <= bits <= 8:
        raise ValueError("discard_bits must be an integer in [0, 8]")
    if bits == 0:
        yield from compatible_blocks(lower, upper, block_length, start)
        return

    low, high, length, first = _inputs(lower, upper, block_length, start)
    width = 1 << bits
    digit_max = 256 // width - 1
    # For a row with code sum Q, every integer total from w*Q through
    # w*Q + n*(w-1) is attainable. A cell exists iff these meet [low, high].
    slack = length * (width - 1)
    code_low = np.maximum(0, -((slack - low) // width))
    code_high = np.minimum(length * digit_max, high // width)
    initial_codes = None if first is None else np.right_shift(first, bits)
    for codes in _walk(code_low, code_high, length, digit_max, initial_codes):
        yield _representative(codes, width, low, high, first)
