//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_OUTPUTLAYER_H
#define NEURALNETLIBRARY_OUTPUTLAYER_H
#include "../FlatteningLayer/FlatteningLayer.h"


namespace cn {
    class OutputLayer : public FlatteningLayer {
        const Bitmap<double> *target;
    public:
        OutputLayer(int id, Vector3<int> _inputSize);
        OutputLayer(const cn::JSON &json);
        void CPURun(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &input) override;
        void setTarget(const Bitmap<double> *_target);
        JSON jsonEncode() const override;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
    };
}

#endif //NEURALNETLIBRARY_OUTPUTLAYER_H
