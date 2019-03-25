#include <cstring>
#include <iostream>

#include "structs/tensor.h"

Tensor::Tensor(int d, int *_n, const T *_data) {
    fill(d, _n, _data);
}

void Tensor::reshape(int _d, int *_n) {
    delete []n;
    n = new int[_d];
    memcpy(n, _n, _d * sizeof(int));
    d = _d;
}

Tensor::T Tensor::get(int *indexes) {
    int idx = indexes[0];
    for (int i = 1; i < d; ++i) {
        idx = idx * n[i] + indexes[i];
    }
    return data[idx];
}

void Tensor::operator =(Tensor t) {
    delete []n;
    delete []data;
    fill(t.d, t.n, t.data);
}

void Tensor::fill(int _d, int *_n, const T *_data) {
    d = _d;
    n = new int[d];
    memcpy(n, _n, d * sizeof(int));
    int sz = 1;
    for (int i = 0; i < d; ++i) {
        sz *= n[i];
    }
    data = new T[sz];
    if (_data == nullptr) {
        memset(data, 0, sz * sizeof(T));
    } else {
        memcpy(data, _data, sz * sizeof(T));
    }
}

Matrix Tensor::getKthMatrix(int k) {
    int lsz = 1;
    int rsz = 1;
    for (int i = 0; i < d; ++i) {
        if (i < k) lsz *= n[i];
        else rsz *= n[i];
    }
    return Matrix(lsz, rsz, data);
}

void Tensor::read(std::istream &is) {
    _read(is, 0, 0);
}

void Tensor::_read(std::istream &is, int curIdx, int pos) {
    if (pos == d - 1) {
        for (int i = 0; i < n[pos]; ++i) {
            is >> *(data + curIdx + i);
        }
        return;
    }
    for (int i = 0; i < n[pos]; ++i) {
        _read(is, (curIdx + i) * n[pos + 1], pos + 1);
    }
}

void Tensor::_write(std::ostream &os, int curIdx, int pos) {
    os << "----- " << pos + 1 << std::endl;
    if (pos == d - 1) {
        for (int i = 0; i < n[pos]; ++i) {
            os << *(data + curIdx + i) << " ";
        }
        os << std::endl;
        return;
    }
    for (int i = 0; i < n[pos]; ++i) {
        _write(os, (curIdx + i) * n[pos + 1], pos + 1);
    }
}

std::ostream& operator << (std::ostream &os, Tensor &t) {
    t._write(os, 0, 0);
    return os;
}
