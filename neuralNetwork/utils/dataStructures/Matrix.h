#pragma once

#include <vector>
class Tensor;
class Matrix{
public:
    friend class Tensor;
    Matrix(int sizeX, int sizeY);
    void edit(int x, int y, float val);
    float getValue(int x, int y) const ;
    int getX() const;
    int getY() const;
private:
    int x,y;
    std::vector< std::vector< float > > values;
};