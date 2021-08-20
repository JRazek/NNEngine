//
// Created by user on 20.08.2021.
//

#include "MaxPoolingLayer.h"

cn::MaxPoolingLayer::MaxPoolingLayer(int _id, int _kernelSizeX, int _kernelSizeY, cn::Network *_network):
        Layer(_id, _network),
        kernelSizeX(_kernelSizeX),
        kernelSizeY(_kernelSizeY){
    //todo init output
}
