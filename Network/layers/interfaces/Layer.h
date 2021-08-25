//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_LAYER_H
#define NEURALNETLIBRARY_LAYER_H

#include <vector>
#include <optional>
#include "../../../Utils/Bitmap.h"

template<typename T>
struct Vector3;
namespace cn {

    class Network;
    class Layer {
    protected:
        Network *network;
        std::optional<Bitmap<float>> output;
        int __id;

        //Bitmap<bool> memoizedChainsStates;
        //Bitmap<float> memoizedChains;
    public:
        Layer(int _id, Network &_network);

        const Bitmap<float> *getOutput() const;
        virtual void run(const Bitmap<float> &bitmap) = 0;

        Layer(const Layer &other) = delete;

        virtual float getChain(const Vector3<int> &input) = 0;

        virtual ~Layer() = default;
        int id() const;
    };
}


#endif //NEURALNETLIBRARY_LAYER_H
