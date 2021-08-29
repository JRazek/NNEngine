//
// Created by user on 29.08.2021.
//

#ifndef NEURALNETLIBRARY_PREFIXSUM2DARR_H
#define NEURALNETLIBRARY_PREFIXSUM2DARR_H
#include "Bitmap.h"
template<typename T>
class PrefixSum2D : public cn::Bitmap<T>{
public:
    template<typename Y>
    PrefixSum2D(const cn::Bitmap<Y> &input);
    T sum(const Vector3<int> &nw, const Vector3<int> &se);
};

template<typename T>
template<typename Y>
PrefixSum2D<T>::PrefixSum2D(const cn::Bitmap<Y> &input):cn::Bitmap<T>(input.size()) {
    for(int z = 0; z < input.d(); z++){
        for(int y = 0; y < input.h(); y++){
            for(int x = 0; x < input.w(); x++){
                if(x == 0 && y == 0){
                    this->setCell(x, y, z, input.getCell(x, y, z));
                }else if(x == 0){
                    this->setCell(x, y, z, this->getCell(x, y - 1, z) + input.getCell(x, y, z));
                }else if(y == 0){
                    this->setCell(x, y, z, this->getCell(x - 1, y, z) + input.getCell(x, y, z));
                }else{
                    this->setCell(x, y, z, this->getCell(x, y - 1, z) + this->getCell(x - 1, y, z) + this->getCell(x - 1, y - 1, z) + input.getCell(x, y, z));
                }
            }
        }
    }
}
template<typename T>
T PrefixSum2D<T>::sum(const Vector3<int> &nw, const Vector3<int> &se) {
    if(nw.x == 0 && nw.y == 0){
        return this->getCell(se);
    }else if(nw.x == 0){
        return this->getCell(se) - this->getCell(se.x, nw.y -1, se.z);
    }else if(nw.y == 0){
        return this->getCell(se) - this->getCell(nw.x -1, se.y, se.z);
    }else{
        return this->getCell(se) - this->getCell(nw.x -1, se.y, se.z) - this->getCell(nw.y -1, se.x, se.z) + this->getCell(nw.x -1, nw.y -1, se.z);
    }
}


#endif //NEURALNETLIBRARY_PREFIXSUM2DARR_H
