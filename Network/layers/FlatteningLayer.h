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
        void run(const Bitmap<float> &input) override;
        virtual float getChain(const Vector3<int> &inputPos) override;
    };
}
#endif //NEURALNETLIBRARY_FLATTENINGLAYER_H
