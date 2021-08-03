//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include "Layer.h"

class Layer;

class FFLayer : public Layer{
public:
    FFLayer(int id, Network * network, int inputSize);
    virtual void run(Bitmap *bitmap) override;
};


#endif //NEURALNETLIBRARY_FFLAYER_H
