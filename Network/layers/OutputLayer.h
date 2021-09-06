//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_OUTPUTLAYER_H
#define NEURALNETLIBRARY_OUTPUTLAYER_H
#include "FlatteningLayer.h"


namespace cn {
    class OutputLayer : public FlatteningLayer {
        const Bitmap<double> *target;
    public:
        OutputLayer(int id, cn::Network &network);
        Bitmap<double> run(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &input) override;
        void setTarget(const Bitmap<double> *_target);
        JSON jsonEncode() const override;
    };
}

#endif //NEURALNETLIBRARY_OUTPUTLAYER_H
