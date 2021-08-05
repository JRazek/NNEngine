//
// Created by jrazek on 30.07.2021.
//

#ifndef NEURALNETLIBRARY_BITMAP_H
#define NEURALNETLIBRARY_BITMAP_H

typedef unsigned char byte;

template<typename T>
class Bitmap {
private:
    T * dataP;
public:
    const int w, h, d;
    Bitmap(int w, int h, int d);
    Bitmap(int w, int h, int d, const T* data);
    Bitmap(const Bitmap &bitmap);
    ~Bitmap();
    T getByte(int col, int row, int depth);
    void setBye(int col, int row, int depth, T b);
    T * data();
};
#include "Bitmap.cpp"
#endif //NEURALNETLIBRARY_BITMAP_H
