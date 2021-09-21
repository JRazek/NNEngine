//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_CONVOLUTIONLAYER_H
#define NEURALNETLIBRARY_CONVOLUTIONLAYER_H

#include <random>
#include "../interfaces/Learnable.h"


namespace cn {
    class CUDAConvolutionLayer;
    class ConvolutionLayer : public Learnable{
    private:
        friend CUDAConvolutionLayer;
        Vector3<int> kernelSize;
        int kernelsCount;
        Vector2<int> padding;
        Vector2<int> stride;

        std::vector<Bitmap<double>> kernels;
        std::vector<double> biases;//ith corresponds to ith kernel

        double diffWeight(int weightID);
        double diffBias(int biasID);

    public:

        ConvolutionLayer(int _id, Vector3<int> _inputSize, Vector2<int> _kernelSize, int _kernelsCount,
                         Vector2<int> _stride, Vector2<int> _padding);

        ConvolutionLayer(const JSON &json);
        void randomInit(std::default_random_engine &randomEngine) override;
        double getChain(const Vector3<int> &inputPos) override;
        int weightsCount() const override;
        int biasesCount() const override;

        virtual std::vector<double> getWeightsGradient() override;
        std::vector<double> getBiasesGradient() override;

        void setBias(int kernelID, double value) override;
        double getBias(int kernelID) const override;

        void setWeight(int weightID, double value) override;
        double getWeight(int weightID) const override;

        virtual void CUDARun(const Bitmap<double> &_input) override;
        void CPURun(const Bitmap<double> &_input) override;

        void CUDAAutoGrad() override;

        JSON jsonEncode() const override;

        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
    };

}


#endif //NEURALNETLIBRARY_CONVOLUTIONLAYER_H
