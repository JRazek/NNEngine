//
// Created by jrazek on 24.09.2021.
//

#ifndef NEURALNETLIBRARY_RECURRENTLAYER_H
#define NEURALNETLIBRARY_RECURRENTLAYER_H

#include <stack>
#include "../interfaces/Layer.h"

namespace cn {
    class RecurrentLayer : public Layer {
        std::stack<Bitmap<double>> memoryStates;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
        void CPURun(const Bitmap<double> &_input) override;
        double getChain(const Vector3<int> &inputPos) override;
        RecurrentLayer(const JSON &json);

        JSON jsonEncode() const override;
    public:
        RecurrentLayer(int _id, Vector3<int> _inputSize);
    };
}

#endif //NEURALNETLIBRARY_RECURRENTLAYER_H
