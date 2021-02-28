#include "Tensor.h"
Tensor::Tensor(int x, int y, int z){
    this->x = x;
    this->y = y;
    this->z = z;
    for(int i = 0; i < z; i ++){
        matrices.push_back(Matrix(x, y));
    } 
}
Tensor::Tensor(int z){
    this->z = z;
}
