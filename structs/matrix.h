#pragma once
#include <iostream>
#include <cmath>

class Tensor;

class Matrix {
public:
    using T = double;

    int n, m;

    Matrix(int n, int m, const T *_data = nullptr);

    Matrix(const Matrix &m) : Matrix(m.n, m.m, m.data) {}

    Matrix(const Matrix &matr, int k, int* idxes);

    void operator =(Matrix m);

    T& operator ()(int i1, int i2) const {
        return data[i1 * m + i2];
    }

    Matrix operator *(Matrix) const;

    Matrix inverse() const;

    Matrix transpose() const;

    void swapLines(int l1, int l2);

    Tensor toTensor(int d, int *n);

    ~Matrix() {
        if (data != nullptr) {
            delete []data;
            data = nullptr;
        }
    }
    friend std::ostream& operator << (std::ostream &os, Matrix m) {
        os << '(' << m.n << ", " << m.m << ')' << std::endl;
        for (int i = 0; i < m.n; ++i) {
            for (int j = 0; j < m.m; ++j) {
                if (fabs(m(i, j)) < 1e-14) os << "0 ";
                else os << m(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }
private:
    void fill(int _n, int _m, const T *data);
    T *data = nullptr;
};
