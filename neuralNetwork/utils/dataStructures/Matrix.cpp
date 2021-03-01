#include "Matrix.h"

Matrix::Matrix(int sizeX, int sizeY){
    this->x = sizeX;
    this->y = sizeY;
    for(int y = 0; y < sizeY; y++){
        values.push_back(std::vector<float>());
        for(int x = 0; x < sizeX; x++){
            values[y].push_back(0);
        }
    }
}
void Matrix::edit(int x, int y, float val){
    values[y][x] = val;
}
float Matrix::getValue(int x, int y) const{
    return values[y][x];
}
int Matrix::getX() const{
    return x;
}
int Matrix::getY() const{
    return y;
}