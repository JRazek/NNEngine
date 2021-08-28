//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_CONVOLUTIONLAYER_H
#define NEURALNETLIBRARY_CONVOLUTIONLAYER_H

#include "interfaces/Learnable.h"

template<typename T>
struct Vector3;
namespace cn {
    class ConvolutionLayer : public Learnable{
    private:
        Vector3<int> kernelSize;
        int kernelsCount;
        int paddingX;
        int paddingY;
        int strideX;
        int strideY;

        const DifferentiableFunction &activationFunction;

        std::vector<Bitmap<float>> kernels;
        std::vector<float> biases;//ith corresponds to ith kernel

        std::optional<cn::Bitmap<float>> beforeActivation;

        float diffWeight(int weightID);

    public:
        ConvolutionLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY,
                         int _kernelsCount, const DifferentiableFunction &_activationFunction,
                         int _paddingX, int _paddingY, int _strideX, int _strideY);
        void randomInit() override;
        Bitmap<float> run(const Bitmap<float> &input) override;
        float getChain(const Vector3<int> &inputPos) override;
        int weightsCount() const override;

        virtual std::vector<float> getGradient() override;

        void setWeight(int weightID, float value) override;
        float getWeight(int weightID) const override;
    };
}


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
