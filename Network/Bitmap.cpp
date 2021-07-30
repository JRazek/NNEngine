//
// Created by jrazek on 30.07.2021.
//

#include "Bitmap.h"
#include <algorithm>
Bitmap::Bitmap(int w, int h, int d): w(w), h (h), d(d) {
    this->data = new byte [w * h * d];
}

Bitmap::Bitmap(const Bitmap &bitmap): Bitmap(bitmap.w, bitmap.w, bitmap.h) {
    std::copy(bitmap.data, bitmap.data + w * h * d, this->data);
}

byte Bitmap::getByte(int col, int row, int depth) {
    if(col >= this->w or col < 0 or row >= this->h or row < 0 or depth >= this->d or depth < 0){
        throw std::invalid_argument("byte does not belong to bitmap!");
    }
    return this->data[depth * w * h + row * w + col];
}

void Bitmap::setBye(int col, int row, int depth, byte b) {
    if(col >= this->w or col < 0 or row >= this->h or row < 0 or depth >= this->d or depth < 0){
        throw std::invalid_argument("wrong byte to set!");
    }
    this->data[depth * w * h + row * w + col] = b;
}
