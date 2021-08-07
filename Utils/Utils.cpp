#include "Bitmap.h"
#include "Utils.h"
//
// Created by jrazek on 05.08.2021.
//

cn::Bitmap<float> cn::Utils::normalize(const Bitmap<byte> &input) {
    Bitmap<float> bitmap (input.w, input.h, input.d);
    for(int i = 0; i < input.w * input.h * input.d; i ++){
        bitmap.data()[i] = 1.f / (256.f -((float)input.data()[i]));
    }
    return bitmap;
}




