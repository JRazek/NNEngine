//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include "Layer.h"

namespace cn {
    class Layer;
    class Network;
    class FFLayer : public cn::Layer{
    public:
        FFLayer(int id, Network * network, int inputSize);
        void run(Bitmap<float> &bitmap) override;
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
