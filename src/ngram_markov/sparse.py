import array
from scipy.sparse import csr_array

import array
import numpy as np


class IncrementalCSRMatrix:

    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.indptr = array.array('i', [0])
        self.indices = array.array('i')
        self.data = array.array(type_flag)

        self.current_row = -1

    def append_row(self, row, row_idx):

        m, n = self.shape

        if row_idx >= m:
            raise Exception('Row index out of bounds')

        if len(row) != n:
            raise Exception('Length of row must match the number of columns')

        self.update_indptr(row_idx)

        nonzero_indices = np.nonzero(row)[0]
        nonzero_values = row[nonzero_indices]

        self.indices.extend(nonzero_indices)
        self.data.extend(nonzero_values)

        self.indptr[row_idx + 1] = len(self.data)
        self.current_row = row_idx

    def update_indptr(self, row_idx):
        if row_idx > self.current_row:
            num_empty_rows = row_idx - self.current_row
            self.indptr.extend([len(self.data)] * num_empty_rows)

    def tocsr(self):

        self.update_indptr(self.shape[0] - 1)

        indptr = np.frombuffer(self.indptr, dtype=np.int32)
        indices = np.frombuffer(self.indices, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return csr_array(
            (data, indices, indptr),
            shape=self.shape
        )

    def __len__(self):
        return len(self.data)
