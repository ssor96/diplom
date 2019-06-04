#!/usr/bin/python3
# -*- coding: utf8 -*-

import time

import numpy as np
import numpy.linalg as la
from sympy import Matrix

from functools import lru_cache

seed = int(time.time())
print('start with seed', seed)
np.random.seed(seed)

enable_trace = 1
call_id = 0
stack = []
def trace(f):
    def g(*args, **kwargs):
        if not enable_trace:
            return f(*args, **kwargs)
        global call_id, stack
        stack.append((f.__name__, call_id))
        print('start', *stack[-1])
        call_id += 1
        res = f(*args, **kwargs)
        print('end', *stack[-1])
        del stack[-1]
        if len(stack):
            print('continue', *stack[-1])
        return res
    return g


# def rand_tensor(*shape):
#     return np.random.rand(*shape)
#
#
# def ones_tensor(*shape):
#     return np.ones(shape)
#
#
# def zeros_tensor(*shape):
#     return np.zeros(shape)
#
#
# def fro(matrix):
#     return la.norm(matrix, 'fro')
#
#
# def dist(m1, m2):
#     return fro(m1 - m2)


@trace
def fold_matrix(matrix, shape):
    return np.reshape(matrix, shape)


@trace
def zeros_tt(*shape):
    cores = list()
    cores.append(np.zeros((shape[0], 1)))
    for n in shape[1:-1]:
        cores.append(np.zeros((1, n, 1)))
    cores.append(np.zeros((1, shape[-1])))
    return cores


@trace
def from_tt_cores(cores):
    cores_iterator = iter(cores)
    tensor = next(cores_iterator)
    for core in cores_iterator:
        # Dot product over rank indexes
        # of tensor of size (n_1, ..., n_{i-1}, r_{i-1})
        # and core of size (r_{i-1}, n_i, r_i)
        # Result is a tensor of size (n_1, ..., n_{i-1}, n_i, r_i)
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    return tensor


@trace
def gen_rand_set(r, n):
    if n < r:
        raise Exception("Too big r")
    if 2 * r > n:
        exclude = set(gen_rand_set(n - r, n))
        return [i for i in range(n) if i not in exclude]
    l = []
    for _ in range(r):
        c = np.random.randint(0, n - 1)
        while c in l:
            c = np.random.randint(0, n - 1)
        l.append(c)
    return sorted(l)


@trace
def get_linear_independent_rows(_a, EPS=1e-9):
    """
    return indexes of linear independent rows and columns
    """

    matr, rows_indexes = Matrix(_a).rref()
    columns_indexes = []
    for r in range(len(rows_indexes)):
        for i in range(matr.shape[1]):
            if abs(matr[r, i]) > 1e-1:
                columns_indexes.append(i)
                break
    #     print(matr, matr.shape)
    return np.array(rows_indexes), np.array(columns_indexes)


@trace
def find_good_submatrix(a, r=1, final=False):
    """
    returns indexes of submatrix columns with max rank (n x r) and linear independent rows in it
    """
    mx_r = 0
    mx_columns_idx = []
    mx_rows_idx = []
    n, m = a.shape
    mx_det = 0
    for _ in range(2):  # try other values
        choosen_columns = gen_rand_set(r, m)
        small_matr = np.array([[a[i, j] for j in choosen_columns] for i in range(n)])
        if np.count_nonzero(small_matr) == 0:
            continue
        rows_idx, columns_idx = get_linear_independent_rows(small_matr)
        if rows_idx is None or len(rows_idx) == 0:
            rows_idx = np.array([0])
            columns_idx = np.array([0])
        temp = small_matr[rows_idx[:, None], columns_idx]
        #         print(rows_idx, columns_idx)
        #         print(temp)
        cur_det = abs(la.det(temp))
        if len(rows_idx) > mx_r or len(rows_idx) == mx_r and cur_det > mx_det:
            mx_r = len(rows_idx)
            mx_det = cur_det
            mx_columns_idx = [choosen_columns[x] for x in columns_idx]
            mx_rows_idx = rows_idx[:]
        if mx_r == r: break
    if mx_r > 0 and mx_det < 1e-6:
        return find_good_submatrix(a, mx_r - 1, final=True)
    if mx_r == r and r < min(n, m) and not final:
        return find_good_submatrix(a, min(2 * r, min(n, m)))
    return mx_rows_idx, mx_columns_idx


@trace
def cross_matrix_approx(a, delta=0.01):
    """
    return rows and columns indexes for approximation
    """
    n = a.shape[0]
    lir, columns = find_good_submatrix(a)  # lir = Linear Independent Rows
    r = len(columns)  # rank
    if r == 0:
        return [], []
    submatr = np.array([[a[i, j] for j in columns] for i in range(n)])
    order = list(range(n))
    cur = 0
    for idx in lir:
        if idx != cur:
            submatr[[idx, cur]] = submatr[[cur, idx]]
            order[idx], order[cur] = order[cur], order[idx]
        cur += 1
    cnt = 0
    while True:
        cnt += 1
        inter_matr = submatr[list(range(r))]
        inv = la.inv(inter_matr)
        print(la.det(inter_matr))
        b = submatr @ la.inv(inter_matr)
        mn = b.min()
        mx = b.max()
        if abs(mn) > abs(mx):
            mx = mn
        wi, wj = map(lambda x: x[0], np.where(b == mx))
        if abs(mx) < 1 + delta:
            break
        print(wi, wj, mx, r)
        submatr[[wi, wj]] = submatr[[wj, wi]]
        order[wi], order[wj] = order[wj], order[wi]
    rows = np.array(sorted(order[:r]))
    return rows, columns


class BlackBoxMatrix:
    def __init__(self, f, n, m):
        self._calls = 0
        self._f = f
        self._n = n
        self._m = m

    def __repr__(self):
        return 'matr {0}x{1}'.format(self._n, self._m)

    @property
    def shape(self):
        return self._n, self._m

    def __getitem__(self, item):
        self._calls += 1
        return self._f(*item)

    @property
    def calls_stat(self):
        return self._calls

@trace
def tt_cross(shape, f):
    """
    Decompose tensor of size (n1, n2, ..., nd) to
    list of TT cores of sizes
    [(n1, r1), (r1, n2, r2), (r2, n2, r3), ..., (rd, nd)]
    """
    decomposition = []

    calls_total = 0
    rows_indexes = [(x,) for x in range(shape[0])]
    cols_num = np.product(shape[1:])
    rows_num = shape[0]
    cur_idx = 0
    while True:
        print(cur_idx, rows_num, cols_num)
        def g(rows, cols):
            cols_tuple = tuple()
            for i in range(cur_idx + 1, len(shape)):
                cols_tuple += (cols % shape[i],)
                cols //= shape[i]
            return f(*rows_indexes[rows], *cols_tuple)
        mtrx = BlackBoxMatrix(g, rows_num, cols_num)
        rows_idx, cols_idx = cross_matrix_approx(mtrx)
        if len(rows_idx) == 0:
            return zeros_tt(*shape)
        u = np.array([[mtrx[i, j] for j in cols_idx] for i in range(rows_num)])
        intersection_mtrx = np.array([[mtrx[i, j] for j in cols_idx] for i in rows_idx])
        # print(intersection_mtrx)
        decomposition.append(u @ la.inv(intersection_mtrx))

        if cur_idx + 1 == len(shape) - 1:
            kernel = np.array([[mtrx[i, j] for j in range(cols_num)] for i in rows_idx])
            decomposition.append(kernel)
            calls_total += mtrx.calls_stat
            break

        calls_total += mtrx.calls_stat
        rows_indexes = [rows_indexes[x] + (i, ) for x in rows_idx for i in range(shape[cur_idx + 1])]
        cols_num //= shape[cur_idx + 1]
        rows_num = len(rows_indexes)
        cur_idx += 1

    for i in range(1, len(decomposition) - 1):
        # Transform (r_{i-1} * n_i, r_i) matrix
        # to (r_{i-1}, n_i, r_i) tensor
        prev_r = decomposition[i - 1].shape[-1]
        r = decomposition[i].shape[1]
        n = decomposition[i].shape[0] // prev_r
        u = fold_matrix(decomposition[i], (prev_r, n, r))
        decomposition[i] = u

    print('done in {} calls'.format(calls_total))
    return decomposition


@trace
def gen_rand_interval_system(n):
    up = [np.random.rand() * 10 for _ in range(n * n)]
    down = [np.random.rand() * 10 for _ in range(n * n)]
    for i in range(n * n):
        down[i], up[i] = sorted((down[i], up[i]))
    up_x = [np.random.rand() * 10 for _ in range(n * n)]
    down_x = [np.random.rand() * 10 for _ in range(n * n)]
    up_b = [0] * n
    down_b = [0] * n
    for i in range(n):
        down_x[i], up_x[i] = sorted((down_x[i], up_x[i]))
        down_b[i] = 0
        up_b[i] = 0
        for j in range(n):
            down_b[i] += down[i * n + j]
            up_b[i] += up[i * n + j]
    return (down, up), (down_b, up_b)

calls = 0
@trace
def solve(A, b):
    """
    solve Ax = b
    """
    global calls
    calls += 1
    # print(A, b)
    return tuple(la.solve(A, b))


n = 3
matr, b = gen_rand_interval_system(n)
@lru_cache(maxsize=10**3)
# @trace
def f_for_cache(A, b, n):
    return solve(np.array(A).reshape((n, n)), np.array(b))


@trace
def f(*args):
    A = tuple(matr[args[i + 1]][i] for i in range(n * n))
    concrete_b = tuple(b[args[n * n + i + 1]][i] for i in range(n))
    return f_for_cache(A, concrete_b, n)[args[0]]


@trace
def main():
    begin = time.time()
    cores = tt_cross([n] + [2] * (n * n + n), f)
    total = time.time() - begin
    print('real calls', calls)
    # print(cores)
    for c in cores:
        print(c.shape)
    print('done in {} sec'.format(total))

    # print(from_tt_cores(cores))


if __name__ == '__main__':
    main()
