//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H
#define NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H

#include "interfaces/Layer.h"
namespace cn {
    class BatchNormalizationLayer : public Layer {

        float normalizationFactor;
    public:
        BatchNormalizationLayer(int _id, Network &_network);
        void run(const Bitmap<float> &bitmap) override;
        float getChain(int neuronID) override;
    };
}
#endif //NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H
