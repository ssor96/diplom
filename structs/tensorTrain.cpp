#include <iostream>
#include <cstring>
#include "structs/tensorTrain.h"
#include "algos/matrixAlgo.h"

TensorTrain::TensorTrain(Tensor t) {
    d = t.d;
    n = new int[d];
    memcpy(n, t.n, d * sizeof(int));
    cores = new Tensor[d];
    int newD = d;
    int prv = 1;
    int coreDims[3];
    for (int i = 0; i < d - 1; ++i) {
        auto [b, c] = skeletonDecomposition(t.getKthMatrix(1 + !!i));
        coreDims[0] = prv;
        coreDims[1] = b.n / prv;
        coreDims[2] = b.m;
        prv = b.m;
        cores[i] = b.toTensor(3, coreDims);
        n[i] = b.m;
        newD -= !!i;
        t = c.toTensor(newD, n + i);
        n[i] = t.n[i];
    }
    Matrix last = t.getKthMatrix(1);
    coreDims[0] = last.n;
    coreDims[1] = last.m;
    coreDims[2] = 1;
    cores[d - 1] = last.toTensor(3, coreDims);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << cores[i].n[j] << " ";
        }
        std::cout << std::endl;
    }
}

TensorTrain::T TensorTrain::get(int *indexes) {
    return 0;
}