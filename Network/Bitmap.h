//
// Created by jrazek on 30.07.2021.
//

#ifndef NEURALNETLIBRARY_BITMAP_H
#define NEURALNETLIBRARY_BITMAP_H

typedef unsigned char byte;
class Bitmap {
private:
    //format is following --
    byte * data;
public:
    const int w, h, d;
    Bitmap(int w, int h, int d);
    Bitmap(const Bitmap &bitmap);
    byte getByte(int col, int row, int depth);
    void setBye(int col, int row, int depth, byte b);
};


#endif //NEURALNETLIBRARY_BITMAP_H
