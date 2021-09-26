//
// Created by jrazek on 09.09.2021.
//

#ifndef NEURALNETLIBRARY_INPUTLAYER_H
#define NEURALNETLIBRARY_INPUTLAYER_H
#include "../interfaces/Layer.h"
namespace cn {

    class InputLayer : public Layer{
        std::vector<Tensor<double>> input;
    public:
        InputLayer(int _id, Vector3<int> _inputSize);
        InputLayer(const JSON &json);
        void CPURun(const Tensor<double> &_input) override;
        virtual double getChain(const Vector4<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        const Tensor<double> &getInput(int time) const override;
        std::unique_ptr <Layer> getCopyAsUniquePtr() const override;

        InputLayer(const InputLayer &inputLayer);
    };

}
#endif //NEURALNETLIBRARY_INPUTLAYER_H
