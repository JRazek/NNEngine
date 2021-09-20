//
// Created by user on 17.09.2021.
//

#include "CUDAConvolutionLayer.cuh"
#include "ConvolutionLayer.h"
#include <memory>
#include "../../../CUDA/CUDAUtils.cuh"

cn::Bitmap<double> cn::CUDAConvolutionLayer::CUDARun(cn::ConvolutionLayer &convolutionLayer, const cn::Bitmap<double> &_input) {
    convolutionLayer.output = std::make_unique<Bitmap<double>>(
            CUDAUtils::cudaConvolve(
                convolutionLayer.kernels, _input,
                convolutionLayer.padding.x, convolutionLayer.padding.y,
                convolutionLayer.stride.x, convolutionLayer.stride.y
            )
    );

    return *convolutionLayer.output.get();
}

void cn::CUDAConvolutionLayer::CUDAAutoGrad(cn::ConvolutionLayer &convolutionLayer) {

}
