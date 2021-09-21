//
// Created by user on 17.09.2021.
//

#ifndef NEURALNETLIBRARY_CUDACONVOLUTIONLAYER_CUH
#define NEURALNETLIBRARY_CUDACONVOLUTIONLAYER_CUH

namespace cn {

    class ConvolutionLayer;
    template<typename T>
    class Bitmap;

    class CUDAConvolutionLayer{
    public:
        static Bitmap<double> CUDARun(ConvolutionLayer &convolutionLayer, const Bitmap<double> &_input);
        static void CUDAAutoGrad(ConvolutionLayer &convolutionLayer);
    };
}



#endif //NEURALNETLIBRARY_CUDACONVOLUTIONLAYER_CUH
