//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H
#define NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H

#include "interfaces/Layer.h"
namespace cn {
    class BatchNormalizationLayer : public Layer {

        double normalizationFactor;
    public:
        BatchNormalizationLayer(int _id, Vector3<int> _inputSize);
        BatchNormalizationLayer(const JSON &json);
        void run(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &inputPos) override;
        JSON jsonEncode() const override;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
    };
}
#endif //NEURALNETLIBRARY_BATCHNORMALIZATIONLAYER_H
