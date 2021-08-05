//
// Created by jrazek on 05.08.2021.
//

#include "Utils.h"

cn::Bitmap<float> *cn::Utils::normalize(const Bitmap<unsigned char> &input) {
    Bitmap<float> * bitmap = new Bitmap<float>(input.w, input.h, input.d);
    for(int i = 0; i < input.w * input.h * input.d; i ++){
        bitmap->data()[i] = 1.f / (256 - input.data()[i]);
    }
    return bitmap;
}