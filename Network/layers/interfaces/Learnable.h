//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H

#include <random>
#include "Layer.h"

namespace cn {
    class Network;
    class Learnable : public Layer {
    protected:

    public:
        virtual std::vector<double *> getBiasesByRef() = 0;
        virtual std::vector<double *> getWeightsByRef() = 0;

        virtual int weightsCount() const = 0;
        virtual int biasesCount() const = 0;
        virtual void setWeight(int weightID, double value) = 0;
        virtual double getWeight(int weightID) const = 0;
        virtual void setBias(int neuronID, double value) = 0;
        virtual double getBias(int neuronID) const = 0;

        virtual void randomInit(std::default_random_engine &randomEngine) = 0;

        Learnable(Vector3<int> _inputSize);
        virtual std::vector<double> getWeightsGradient() = 0;
        virtual std::vector<double> getBiasesGradient() = 0;
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
