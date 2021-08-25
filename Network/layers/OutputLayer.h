//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_OUTPUTLAYER_H
#define NEURALNETLIBRARY_OUTPUTLAYER_H
#include "FlatteningLayer.h"


namespace cn {
    class OutputLayer : public FlatteningLayer {
    public:
        OutputLayer(int id, cn::Network &network);
        void run(const Bitmap<float> &input) override;
        float getChain(const Vector3<float> &input) override;

    };
}

#endif //NEURALNETLIBRARY_OUTPUTLAYER_H
