//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H


#include "Layer.h"

class FFLayer : public Layer{
public:
    FFLayer(int inputSize);
    virtual void run() override;
};


#endif //NEURALNETLIBRARY_FFLAYER_H
