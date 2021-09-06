//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_MAXPOOLINGLAYER_H
#define NEURALNETLIBRARY_MAXPOOLINGLAYER_H
#include "interfaces/Layer.h"

namespace cn {
    class MaxPoolingLayer : public Layer{
    public:
        Vector2<int> kernelSize;

        std::optional<Bitmap<Vector2<int>>> mapping;

        MaxPoolingLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY);
        Bitmap<double> run(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &inputPos) override;
        JSON jsonEncode() const override;
    };
}

#endif //NEURALNETLIBRARY_MAXPOOLINGLAYER_H
