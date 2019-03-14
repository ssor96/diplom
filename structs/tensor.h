#pragma once
#include <iostream>
#include "structs/matrix.h"

class Tensor {
public:
    using T = double;
    int d;

    Tensor(int d, int *_n, const T *_data = nullptr);

    Tensor(Tensor &t) : Tensor(t.d, t.n, t.data) {}

    void operator = (Tensor t);

    void reshape(int k, int *_n);

    T& get(int *indexes);

    void read(std::istream &is);

    Matrix getKthMatrix(int k);

    friend std::ostream& operator << (std::ostream &os, Tensor &t);

    ~Tensor() {
        delete []n;
        if (data != nullptr) {
            delete []data;
        }
    }
private:
    int *n;
    T *data = nullptr;
    void fill(int _d, int *_n, const T *_data);
    void _read(std::istream &is, int curIdx, int pos);
    void _write(std::ostream &is, int curIdx, int pos);
};
