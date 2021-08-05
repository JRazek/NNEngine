//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_CONVOLUTIONLAYER_H
#define NEURALNETLIBRARY_CONVOLUTIONLAYER_H

#include "Layer.h"

class ConvolutionLayer : public Layer{
public:
    static Bitmap<float> * convolve(const Bitmap<float> *kernel, const Bitmap<float> *input, int paddingX, int paddingY, int stepX, int stepY);
    static int afterConvolutionSize(int kernelSize, int inputSize, int padding, int step);
    ConvolutionLayer(int id, Network *network, int kernelsCount);
    void run(Bitmap<float> *bitmap) override;
};


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
