//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_OUTPUTLAYER_H
#define NEURALNETLIBRARY_OUTPUTLAYER_H
#include "../FlatteningLayer/FlatteningLayer.h"


namespace cn {
    class CUDAOutputLayer;

    class OutputLayer : public FlatteningLayer {
        friend CUDAOutputLayer;
        const Tensor<double> *target;
    public:
        OutputLayer(int id, Vector3<int> _inputSize);
        OutputLayer(const cn::JSON &json);
        void CPURun(const Tensor<double> &input) override;
        double getChain(const Vector4<int> &input) override;
        void setTarget(const Tensor<double> *_target);
        JSON jsonEncode() const override;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;

        #ifdef NNL_WITH_CUDA
        void CUDAAutoGrad() override;
        #endif
    };
}

#endif //NEURALNETLIBRARY_OUTPUTLAYER_H
