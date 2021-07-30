//
// Created by jrazek on 27.07.2021.
//

#include "ConvolutionLayer.h"
#include "../Bitmap.h"

void ConvolutionLayer::run() {
    const Bitmap * inputBitmap;
    if(Layer::network->getLayers()->front()->id == ConvolutionLayer::id){
        //im the first layer in network
    }else if(auto * l = dynamic_cast<ConvolutionLayer *>(Layer::network->getLayers()->at(ConvolutionLayer::id - 1))) {
        //get output of prev layer
    }

}

ConvolutionLayer::ConvolutionLayer(int id, Network *network, int w, int h, int d) : Layer(id, network) {

}
