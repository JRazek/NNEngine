//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_CONVOLUTIONLAYER_H
#define NEURALNETLIBRARY_CONVOLUTIONLAYER_H

#include "Layer.h"

class ConvolutionLayer : public Layer{
public:
    static Bitmap * convolve(const Bitmap *kernel, const Bitmap *input, int paddingX, int paddingY, int stepX, int stepY);
    static int afterConvolutionSize(int kernelSize, int inputSize, int padding, int step);
    ConvolutionLayer(int id, Network *network, int kernelsCount);
    void run(Bitmap *bitmap) override;
};


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
