//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_CONVOLUTIONLAYER_H
#define NEURALNETLIBRARY_CONVOLUTIONLAYER_H

#include "Layer.h"

namespace cn {
    class ConvolutionLayer : public cn::Layer{
    private:
        const int kernelSizeX;
        const int kernelSizeY;
        const int kernelSizeZ;
        const int kernelsCount;
        const int paddingX;
        const int paddingY;
        const int strideX;
        const int strideY;

        std::vector<Bitmap<float>> kernels;


    public:
        ConvolutionLayer(int _id, cn::Network *_network, int _kernelSizeX, int _kernelSizeY, int _kernelSizeZ, int _kernelsCount, int _paddingX, int _paddingY,
                         int _strideX, int _strideY);

        void run(const Bitmap<float> &bitmap) override;
    };
}


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
