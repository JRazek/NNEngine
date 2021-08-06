//
// Created by jrazek on 30.07.2021.
//

#ifndef NEURALNETLIBRARY_BITMAP_H
#define NEURALNETLIBRARY_BITMAP_H


namespace cn {
    template<typename T>
    class Bitmap {
    private:
        T * dataP;
    public:
        const int w, h, d;
        Bitmap(int w, int h, int d);
        /**
         *
         * @param w - data width
         * @param h - data height
         * @param d - number of channels
         * @param data - data itself
         * @param options - format of data
         * 0 - for standard (each column and channel is in ascending order)
         * 1 - Ordering pixel on (x, y) pos in each channel is next to each other. Sth like RGB ordering
         */
        Bitmap(int w, int h, int d, const T* data, int options = 0);
        Bitmap(const Bitmap &bitmap);
        ~Bitmap();
        T getByte(int col, int row, int depth);
        void setBye(int col, int row, int depth, T b);
        T * data() const;
    };
}

#include "Bitmap.cpp"


#endif //NEURALNETLIBRARY_BITMAP_H
