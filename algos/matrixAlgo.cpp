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
    int startRow = 0;
    for (int j = 0; j < m.m; ++j) {
        int where = startRow;
        for (int i = startRow + 1; i < m.n; ++i) {
            if (fabs(m(i, j)) > fabs(m(where, j))) {
                where = i;
            }
        }
        if (fabs(m(where, j)) < EPS) {
            continue;
        }
        std::swap(order[startRow], order[where]);
        m.swapLines(startRow, where);
        for (int i = startRow + 1; i < m.n; ++i) {
            double mult = m(i, j) / m(startRow, j);
            m(i, j) = 0;
            for (int k = j + 1; k < m.m; ++k) {
                m(i, k) -= mult * m(startRow, k);
            }
        }
        startRow++;
    }
    order.resize(startRow);
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