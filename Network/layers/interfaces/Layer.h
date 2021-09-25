//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_LAYER_H
#define NEURALNETLIBRARY_LAYER_H

#include <vector>
#include <optional>
#include "../../../Utils/dataStructures/Tensor.h"
#include "../../../Utils/interfaces/JSONEncodable.h"

namespace cn {

    class FFLayer;

    template<typename T>
    struct Vector3;
    class Layer : public JSONEncodable{
    protected:

        std::unique_ptr<Tensor<double>> output;

        Layer *prevLayer = nullptr;
        Layer *nextLayer = nullptr;

        Vector3<int> inputSize;
        Vector3<int> outputSize;

        std::unique_ptr<Tensor<bool>> memoizationStates;
        std::unique_ptr<Tensor<double>> memoizationTable;

        int __id;


    public:
        Layer(int _id, Vector3<int> _inputSize);
        Layer(const Layer &layer);
        Layer(Layer &&layer);

        virtual double getChain(const Vector3<int> &inputPos) = 0;

        /**
         *
         * counts gradient iteratively.
         * @warning Layers must be called in correct order. From the ending layer to the first. Otherwise error will be thrown.
         */
        virtual void CUDAAutoGrad();


        virtual ~Layer() = default;

        void resetMemoization();
        void setMemo(const Vector3<int> &pos, double val);
        bool getMemoState(const Vector3<int> &pos) const;
        double getMemo(const Vector3<int> &pos) const;
        Vector3<int> getOutputSize() const;

        [[maybe_unused]] int id() const;

        virtual JSON jsonEncode() const override;

        virtual std::unique_ptr<Layer> getCopyAsUniquePtr() const = 0;

        static std::unique_ptr<cn::Layer> fromJSON(const cn::JSON &json);

        void setPrevLayer(Layer *_prevLayer);

        const std::unique_ptr<Tensor<double>> &getOutput() const;
        virtual const std::unique_ptr<Tensor<double>> &getInput() const;

        void setNextLayer(Layer *_nextLayer);

        /**
         *
         * @param _input to process
         */
        virtual void CPURun(const Tensor<double> &_input) = 0;

        /**
         * if not supported yet - CPURun is being called.
         */
        virtual void CUDARun(const Tensor<double> &_input);
    };
}


#endif //NEURALNETLIBRARY_LAYER_H
