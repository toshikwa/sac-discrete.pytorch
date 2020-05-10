import operator


class SegmentTree:

    def __init__(self, size, op, init_val):
        assert size > 0 and size & (size - 1) == 0
        self._size = size
        self._op = op
        self._init_val = init_val
        self._values = [init_val for _ in range(2 * size)]

    def _reduce(self, start=0, end=None):
        if end is None:
            end = self._size
        elif end < 0:
            end += self._size

        start += self._size
        end += self._size

        res = self._init_val
        while start < end:
            if start & 1:
                res = self._op(res, self._values[start])
                start += 1

            if end & 1:
                end -= 1
                res = self._op(res, self._values[end])

            start //= 2
            end //= 2

        return res

    def __setitem__(self, idx, val):
        assert 0 <= idx < self._size

        # Set value.
        idx += self._size
        self._values[idx] = val

        # Update its ancestors iteratively.
        idx = idx >> 1
        while idx >= 1:
            left = 2 * idx
            self._values[idx] = \
                self._op(self._values[left], self._values[left + 1])
            idx = idx >> 1

    def __getitem__(self, idx):
        assert 0 <= idx < self._size
        return self._values[idx + self._size]


class SumTree(SegmentTree):

    def __init__(self, size):
        super().__init__(size, operator.add, 0.0)

    def sum(self, start=0, end=None):
        return self._reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1

        # Traverse to the leaf.
        while idx < self._size:
            left = 2 * idx
            if self._values[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self._values[left]
                idx = left + 1
        return idx - self._size


class MinTree(SegmentTree):

    def __init__(self, size):
        super().__init__(size, min, float("inf"))

    def min(self, start=0, end=None):
        return self._reduce(start, end)
