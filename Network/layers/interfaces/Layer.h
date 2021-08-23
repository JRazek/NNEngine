//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_LAYER_H
#define NEURALNETLIBRARY_LAYER_H

#include <vector>
#include <optional>
#include "../../../Utils/Bitmap.h"
namespace cn {

    class Network;

    class Layer {
    protected:
        Network * network;
    public:
        std::optional<Bitmap<float>> output;

        const int id;
        Layer(int _id, Network &_network);

        virtual void run(const Bitmap<float> &bitmap) = 0;
        virtual ~Layer() = default;

        /**
         *
         * @param entryID - in case of FF - neuron, in case of Conv - kernel
         * @return differentiation chain calculated from that neuron
         */
        virtual float getChain(int entryID) = 0;
    };
}


#endif //NEURALNETLIBRARY_LAYER_H
