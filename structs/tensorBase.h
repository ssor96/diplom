#pragma once

class TensorBase {
public:
    using T = double;
    int d;
    virtual T get(int *indexes) = 0;
protected:
    int *n;
};