//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_CONVOLUTIONLAYER_H
#define NEURALNETLIBRARY_CONVOLUTIONLAYER_H

#include "Layer.h"

class ConvolutionLayer : public Layer{
public:
    ConvolutionLayer(int id, Network *network, int w, int h, int d, int kernelsCount);
    virtual void run() override;
};


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
