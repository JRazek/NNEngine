#include "Tensor.h"
#include <stdexcept>
Tensor::Tensor(int x, int y, int z){
    this->x = x;
    this->y = y;
    this->z = z;
    for(int i = 0; i < z; i ++){
        matrices.push_back(Matrix(x, y));
    } 
}
const Matrix &Tensor::getMatrix(int z) const{
    return matrices[z];
}
void Tensor::pushMatrix(Matrix m){
    if(matrices.size() == 0){
        x = m.x;
        y = m.y;
    }else if(x != m.x || y != m.y){
        throw std::invalid_argument( "pushed matrix size must match!");
    }
    matrices.push_back(m);
    z++;
}
Tensor::Tensor(){}
void Tensor::edit(int x, int y, int z, float val){
    this->matrices[z].values[y][x] = val;
}
float Tensor::getValue(int x, int y, int z){
    return this->matrices[z].values[y][x];
}
int Tensor::getX() const{
    return x;
}
int Tensor::getY() const{
    return y;
}
int Tensor::getZ() const{
    return z;
}