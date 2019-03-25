#include <iostream>
#include <vector>
#include "structs/matrix.h"
#include "algos/matrixAlgo.h"
#include "structs/tensor.h"
#include "structs/tensorTrain.h"

int main() {
    int d;
    std::cin >> d;
    int *n = new int[d];
    for (int i = 0; i < d; ++i) {
        std::cin >> n[i];
    }
    Tensor t(d, n);

    t.read(std::cin);

    std::cout << d << std::endl;

    TensorTrain tt(t);

    // std::cout << "PROOF" << std::endl;

    // Matrix check = tmp.getKthMatrix(1);
    // std::cout << check << std::endl;
    // for (int i = v.size() - 1; i >= 0; --i) {
    //     int mlt = check.n * check.m;
    //     check.n = v[i].m;
    //     check.m = mlt / v[i].m;
    //     check = v[i] * check;
    //     std::cout << i << "th Matrix:" << std::endl << v[i] << std::endl << "mult " << i << std::endl << check;
    // }
    // std::cout << "RESULT IS" << std::endl << check << std::endl << t.getKthMatrix(1);
    // std::cout << "Number of params: " << params << " of " << check.n * check.m << std::endl;
    delete []n;
}
