//
// Created by jrazek on 27.07.2021.
//

#include "ConvolutionLayer.h"
#include "../Network.h"
#include "../Bitmap.h"

void ConvolutionLayer::run() {
    Bitmap * inputBitmap;
    if(Layer::network->getLayers()->front()->id == ConvolutionLayer::id){
        const byte * data = this->network->data;
        inputBitmap = new Bitmap(this->network->dataWidth, this->network->dataHeight, this->network->dataDepth, data);
    }else if(auto * l = dynamic_cast<ConvolutionLayer *>(Layer::network->getLayers()->at(ConvolutionLayer::id - 1))) {
        //get output of prev layer
    }

    delete inputBitmap;
}

ConvolutionLayer::ConvolutionLayer(int id, Network *network, int w, int h, int d, int kernelsCount) : Layer(id, network) {

}
