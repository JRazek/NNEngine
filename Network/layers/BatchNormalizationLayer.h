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
        Bitmap<float> run(const Bitmap<float> &input) override;
        float getChain(const Vector3<int> &inputPos) override;
    };
}
#endif //NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H
