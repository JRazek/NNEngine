#pragma once
#include "Matrix.h"
class Tensor{
public:
    Tensor(int x, int y, int z);
    Tensor();
    void edit(int x, int y, int z, float val);
    void pushMatrix(Matrix m);
    float getValue(int x, int y, int z) const;
    const Matrix &getMatrix(int z) const;
    int getX() const;
    int getY() const;
    int getZ() const;
private:
    int x,y,z;
    std::vector<Matrix> matrices;
};