#include "Tensor.h"
Tensor::Tensor(int x, int y, int z){
    for(int i = 0; i < z; i ++){
        matrices.push_back(Matrix(x, y));
    } 
}
