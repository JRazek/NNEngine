//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include "Layer.h"
#include "RandomInitiable.h"

namespace cn {
    class Layer;
    class Network;
    class FFLayer : public cn::Layer, public RandomInitiable{
        std::vector<float> weights;
    public:
        void randomInit() override ;
        FFLayer(int _id, Network * _network, int _inputSize);
        void run(const Bitmap<float> &bitmap) override;
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
