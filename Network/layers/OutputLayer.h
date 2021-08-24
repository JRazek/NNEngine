//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_OUTPUTLAYER_H
#define NEURALNETLIBRARY_OUTPUTLAYER_H
#include "FlatteningLayer.h"

namespace cn {
    class OutputLayer : public FlatteningLayer {
        class Backpropagation;
    protected:
        friend Backpropagation;
        std::optional<Bitmap<float>> target;
    public:
        OutputLayer(int id, cn::Network &network);
        virtual float getChain(int neuronID) override;
    };
}

#endif //NEURALNETLIBRARY_OUTPUTLAYER_H
