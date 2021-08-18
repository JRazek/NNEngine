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
        std::vector<float> weights;
    public:
        void randomInit();
        FFLayer(int _id, Network * _network, int _inputSize);
        void run(const Bitmap<float> &bitmap) override;
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
