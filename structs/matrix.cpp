#include <algorithm>
#include <cstring>
#include <iostream>

#include "structs/tensor.h"
#include "structs/matrix.h"
#include "algos/matrixAlgo.h"

Matrix::Matrix(int n, int m, const T *_data) {
    fill(n, m, _data);
}

Matrix::Matrix(const Matrix &matr, int k, int* idxes) : n(k), m(matr.m) {
    data = new T[n * m];
    for (int i = 0; i < k; ++i) {
        memcpy(data + i * m, matr.data + idxes[i] * m, m * sizeof(T));
    }
}

void Matrix::swapLines(int l1, int l2) {
    std::swap_ranges(data + l1 * m, data + (l1 + 1) * m, data + l2 * m);
}

Matrix Matrix::operator *(Matrix other) const{
    Matrix res(n, other.m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < other.m; ++j) {
            for (int k = 0; k < m; ++k) {
                res(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return res;
}

void Matrix::operator = (Matrix m) {
    if (data != nullptr) {
        delete []data;
    }
    fill(m.n, m.m, m.data);
}

Matrix Matrix::transpose() const {
    Matrix res(m, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            res(j, i) = (*this)(i, j);
        }
    }
    return res;
}

Matrix Matrix::inverse() const {
    Matrix tmp(n, 2 * m);
    for (int i = 0; i < n; ++i) {
        memcpy(tmp.data + 2 * i * m, data + i * m, m * sizeof(T));
        tmp(i, m + i) = 1;
    }

    gauss(tmp);

    for (int i = n - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            T mult = tmp(j, i) / tmp(i, i);
            for (int k = n; k < 2 * m; ++k) {
                tmp(j, k) -= tmp(i, k) * mult;
            }
        }
        for (int j = n; j < 2 * m; ++j) {
            tmp(i, j) /= tmp(i, i);
        }
    }

    Matrix res(n, m);
    for (int i = 0; i < n; ++i) {
        memcpy(res.data + i * m, tmp.data + 2 * i * m + m, m * sizeof(T));
        tmp(i, m + i) = 1;
    }
    return res;
}

void Matrix::fill(int _n, int _m, const T *_data) {
    n = _n;
    m = _m;
    data = new T[n * m];
    if (_data == nullptr) {
        memset(data, 0, n * m * sizeof(T));
    } else {
        memcpy(data, _data, n * m * sizeof(T));
    }
}

Tensor Matrix::toTensor(int d, int *n) {
    return Tensor(d, n, data);
}
