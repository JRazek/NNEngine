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
        Vector2<int> padding;
        Vector2<int> stride;

        std::vector<Bitmap<double>> kernels;
        std::vector<double> biases;//ith corresponds to ith kernel

        double diffWeight(int weightID);
        double diffBias(int biasID);

    public:
        ConvolutionLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY,
                         int _kernelsCount, int _strideX, int _strideY, int _paddingX, int _paddingY);
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

        JSON jsonEncode() const override;
    };

}


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
