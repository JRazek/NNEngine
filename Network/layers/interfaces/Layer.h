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
        std::optional<Bitmap<bool>> memoizationStates;
        std::optional<Bitmap<float>> memoizationTable;

        int __id;

        //Bitmap<bool> memoizedChainsStates;
        //Bitmap<float> memoizedChains;
    public:
        Layer(int _id, Network &_network);

        const Bitmap<float> *getOutput() const;
        virtual void run(const Bitmap<float> &bitmap) = 0;

        Layer(const Layer &other) = delete;

        virtual float getChain(const Vector3<int> &inputPos) = 0;

        virtual ~Layer() = default;

        void resetMemoization();
        void setMemo(const Vector3<int> &pos, float val);
        bool getMemoState(const Vector3<int> &pos) const;
        float getMemo(const Vector3<int> &pos) const;

        [[maybe_unused]] int id() const;
    };
}


#endif //NEURALNETLIBRARY_LAYER_H
