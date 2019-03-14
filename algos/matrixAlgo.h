#pragma once
#include "structs/matrix.h"
#include <vector>

std::vector<int> gauss(Matrix&);

Matrix getLinearIndependentStrings(const Matrix&);

std::pair<Matrix, Matrix> skeletonDecomposition(const Matrix);