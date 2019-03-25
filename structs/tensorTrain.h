#pragma once

#include "structs/tensorBase.h"
#include "structs/tensor.h"

class TensorTrain final : public TensorBase {
public:
    TensorTrain(Tensor t);
    T get(int *indexes) override;
    ~TensorTrain() {
        if (cores != nullptr) {
            delete []cores;
            cores = nullptr;
        }
        if (n != nullptr) {
            delete []n;
            n = nullptr;
        }
    }
private:
    Tensor *cores = nullptr;
};