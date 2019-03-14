#include <iostream>
#include <vector>
#include "structs/matrix.h"
#include "algos/matrixAlgo.h"
#include "structs/tensor.h"

int main() {
    // int n, m;
    // std::cin >> n >> m;
    // double *data = new double[n * m];
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < m; ++j) {
    //         std::cin >> data[i * m + j];
    //     }
    // }
    // Matrix mm(n, m, data);
    // delete []data;
    // auto [b, c] = skeletonDecomposition(mm);

    // std::cout << "------" << std::endl << mm << " = " << std::endl << b << " * " << std::endl << c << std::endl << " ? " << std::endl << b * c << std::endl;

    int d;
    std::cin >> d;
    int *n = new int[d];
    for (int i = 0; i < d; ++i) {
        std::cin >> n[i];
    }
    Tensor t(d, n);

    t.read(std::cin);

    std::cout << d << std::endl;

    Tensor tmp = t;
    int newD = d;
    std::vector<Matrix> v;
    int prv = 1;
    for (int i = 0; i < d - 1; ++i) {
        std::cout << "step " << i << std::endl;
        auto [b, c] = skeletonDecomposition(tmp.getKthMatrix(1 + !!i));
        std::cout << tmp.getKthMatrix(1 + !!i) << std::endl << b << std::endl << c << std::endl << b * c << std::endl;
        std::cout << "cur tensor sizes " << prv << " " << b.n / prv << " " << b.m << std::endl;
        prv = b.m;
        v.push_back(b);
        n[i] = b.m;
        newD -= !!i;
        std::cout << "NEW DIM: ";
        for (int j = 0; j < newD; ++j) {
            std::cout << n[i + j] << " ";
        }
        std::cout << std::endl;
        tmp = c.toTensor(newD, n + i);
    }
    Matrix last = tmp.getKthMatrix(1);
    std::cout << "Last " << d - 1 << std::endl << last << std::endl << "cur tensor sizes " << last.n << " " << last.m << " " << 1 << std::endl;
    // v.push_back(tmp.getKthMatrix(1));

    std::cout << "PROOF" << std::endl;

    Matrix check = tmp.getKthMatrix(1);
    std::cout << check << std::endl;
    for (int i = v.size() - 1; i >= 0; --i) {
        int mlt = check.n * check.m;
        check.n = v[i].m;
        check.m = mlt / v[i].m;
        check = v[i] * check;
        std::cout << i << "th Matrix:" << std::endl << v[i] << std::endl << "mult " << i << std::endl << check;
    }
    std::cout << "RESULT IS" << std::endl << check;
    delete []n;
}
