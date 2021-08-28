//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_OUTPUTLAYER_H
#define NEURALNETLIBRARY_OUTPUTLAYER_H
#include "FlatteningLayer.h"


namespace cn {
    class OutputLayer : public FlatteningLayer {
        const Bitmap<float> *target;
    public:
        OutputLayer(int id, cn::Network &network);
        Bitmap<float> run(const Bitmap<float> &input) override;
        float getChain(const Vector3<int> &input) override;
        void setTarget(const Bitmap<float> *_target);
    };
}

#endif //NEURALNETLIBRARY_OUTPUTLAYER_H
