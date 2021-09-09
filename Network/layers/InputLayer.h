//
// Created by jrazek on 09.09.2021.
//

#ifndef NEURALNETLIBRARY_INPUTLAYER_H
#define NEURALNETLIBRARY_INPUTLAYER_H
#include "interfaces/Layer.h"
namespace cn {

    class InputLayer : public Layer{
        std::optional<Bitmap<double>> input;
    public:
        InputLayer(int _id, Vector3<int> _inputSize);
        InputLayer(const JSON &json);
        void run(const Bitmap<double> &_input) override;
        virtual double getChain(const Vector3<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        const std::optional<Bitmap<double>> &getInput() const override;
        std::unique_ptr <Layer> getCopyAsUniquePtr() const override;
    };

}
#endif //NEURALNETLIBRARY_INPUTLAYER_H
