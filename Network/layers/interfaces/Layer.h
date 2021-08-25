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
        const Bitmap<float> *_input;
        int __id;

        bool chainMemoized;
        float chainValue;
    public:
        Layer(int _id, Network &_network);

        const Bitmap<float> *getOutput();
        virtual void run(const Bitmap<float> &bitmap);

        Layer(const Layer &other) = delete;

        virtual float getChain(const Vector3<float> &input) = 0;

        virtual ~Layer() = default;
        int id() const;
    };
}


#endif //NEURALNETLIBRARY_LAYER_H
