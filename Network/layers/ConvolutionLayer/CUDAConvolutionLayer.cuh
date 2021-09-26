//
// Created by user on 17.09.2021.
//

#ifndef NEURALNETLIBRARY_CUDACONVOLUTIONLAYER_CUH
#define NEURALNETLIBRARY_CUDACONVOLUTIONLAYER_CUH

namespace cn {

    class ConvolutionLayer;
    template<typename T>
    class Tensor;

    class CUDAConvolutionLayer{
    public:
        static void CUDARun(ConvolutionLayer &convolutionLayer, const Tensor<double> &_input);
        static void CUDAAutoGrad(ConvolutionLayer &convolutionLayer);
    };
}



#endif //NEURALNETLIBRARY_CUDACONVOLUTIONLAYER_CUH
