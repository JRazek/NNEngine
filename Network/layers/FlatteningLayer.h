//
// Created by jrazek on 19.08.2021.
//

#ifndef NEURALNETLIBRARY_FLATTENINGLAYER_H
#define NEURALNETLIBRARY_FLATTENINGLAYER_H

#include "interfaces/Layer.h"

namespace cn {
    class Network;
    class FlatteningLayer : public Layer {
    public:
        FlatteningLayer(int _id, Network &_network);
        Bitmap<double> run(const Bitmap<double> &input) override;
        virtual double getChain(const Vector3<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
    };
}
#endif //NEURALNETLIBRARY_FLATTENINGLAYER_H
