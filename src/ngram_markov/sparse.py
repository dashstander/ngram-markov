import array
import scipy.sparse as sp

import array
import numpy as np


class IncrementalCSRMatrix:

    def __init__(self, shape, dtype, index_dtype=np.float32):

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
        
        if index_dtype is np.int32:
            index_type = 'i'
        elif index_dtype is np.int64:
            index_type = 'l'
        else:
            raise Exception('Dtype not supported.')



        self.dtype = dtype
        self.shape = shape

        self.indptr = array.array(index_type, [0])
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

        return sp.csr_array(
            (data, indices, indptr),
            shape=self.shape
        )

    def __len__(self):
        return len(self.data)



class IncrementalCOOMatrix(object):

    def __init__(self, shape, dtype, index_dtype=np.int32):

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
        
        if index_dtype is np.int32:
            index_type = 'i'
        elif index_dtype is np.int64:
            index_type = 'l'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array(index_type)
        self.cols = array.array(index_type)
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if (i >= m or j >= n):
            raise Exception('Index out of bounds')

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def append_row(self, x, idx):

        m, n = self.shape

        if idx >= m:
            raise Exception('Row index out of bounds')

        if x.shape[0] != n:
            raise Exception('Input array has incorrect shape')

        nonzero_indices = np.nonzero(x)[0]

        for j in nonzero_indices:
            self.append(idx, j, x[j])

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return sp.coo_array((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)
