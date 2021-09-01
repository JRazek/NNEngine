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

        std::vector<Bitmap<double>> kernels;
        std::vector<double> biases;//ith corresponds to ith kernel

        std::optional<cn::Bitmap<double>> beforeActivation;

        double diffWeight(int weightID);

    public:
        ConvolutionLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY,
                         int _kernelsCount, const DifferentiableFunction &_activationFunction,
                         int _paddingX, int _paddingY, int _strideX, int _strideY);
        void randomInit() override;
        Bitmap<double> run(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &inputPos) override;
        int weightsCount() const override;
        int biasesCount() const override;


        virtual std::vector<double> getWeightsGradient() override;
        std::vector<double> getBiasesGradient() override;

        void setBias(int kernelID, double value) override;
        double getBias(int kernelID) const override;


        void setWeight(int weightID, double value) override;
        double getWeight(int weightID) const override;
    };
}


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
