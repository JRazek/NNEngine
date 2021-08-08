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
        ConvolutionLayer(int id, cn::Network *network, int kernelSizeX, int kernelSizeY, int kernelSizeZ, int kernelsCount, int paddingX = 0, int paddingY = 0,
                         int strideX = 0, int strideY = 0);

        void run(Bitmap<float> &bitmap) override;

    };
}


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
