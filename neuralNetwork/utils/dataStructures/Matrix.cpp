#include "Matrix.h"

Matrix::Matrix(int sizeX, int sizeY){
    this->x = sizeX;
    this->y = sizeY;
    for(int y = 0; y < sizeY; y++){
        weights.push_back(std::vector<float>());
        for(int x = 0; x < sizeX; x++){
            weights[y].push_back(0);
        }
    }
}