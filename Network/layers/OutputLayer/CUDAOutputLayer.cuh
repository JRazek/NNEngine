//
// Created by user on 21.09.2021.
//

#ifndef NEURALNETLIBRARY_CUDAOUTPUTLAYER_CUH
#define NEURALNETLIBRARY_CUDAOUTPUTLAYER_CUH

namespace cn {
    class OutputLayer;
    class CUDAOutputLayer {
    public:
        static void CUDAAutoGrad(OutputLayer &outputLayer);
    };
}

#endif //NEURALNETLIBRARY_CUDAOUTPUTLAYER_CUH
