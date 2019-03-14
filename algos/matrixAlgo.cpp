#include "algos/matrixAlgo.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

const double EPS = 1e-9;

std::vector<int> gauss(Matrix &m) {
    std::vector<int> order(m.n);
    for (int i = 0; i < m.n; ++i) {
        order[i] = i;
    }
    for (int i = 0; i < m.n; ++i) {
        int where = i;
        for (int j = i + 1; j < m.n; ++j) {
            if (fabs(m(i, j)) > fabs(m(i, j))) {
                where = j;
            }
        }
        if (fabs(m(i, where)) < EPS) {
            order.resize(i);
            break;
        }
        std::swap(order[i], order[where]);
        m.swapLines(i, where);
        for (int j = i + 1; j < m.n; ++j) {
            double mult = m(j, i) / m(i, i);
            m(j, i) = 0;
            for (int k = i + 1; k < m.m; ++k) {
                m(j, k) -= mult * m(i, k);
            }
        }
    }
    return order;
}

Matrix getLinearIndependentStrings(const Matrix &m) {
    Matrix tmp(m);
    std::vector<int> idxes = gauss(tmp);
    return Matrix(m, idxes.size(), idxes.data());
}

std::pair<Matrix, Matrix> skeletonDecomposition(const Matrix m) {
    Matrix res = getLinearIndependentStrings(m);

    Matrix tmp = m * res.transpose() * (res * res.transpose()).inverse();

    return std::make_pair(tmp, res);
}