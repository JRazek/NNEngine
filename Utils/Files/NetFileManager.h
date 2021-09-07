//
// Created by jrazek on 07.09.2021.
//

#ifndef NEURALNETLIBRARY_NETFILEMANAGER_H
#define NEURALNETLIBRARY_NETFILEMANAGER_H
#include "../../Network/layers/interfaces/Layer.h"
#include "../../Network/layers/FFLayer.h"
#include <memory>
namespace cn {
    class NetFileManager {
        std::unique_ptr<Layer> layerFromJSON(Network &network, const JSON &json){
            std::unique_ptr<Layer> uniquePtr = std::make_unique<FFLayer>(1, 1, network);
            return uniquePtr;
        }
    };
}

#endif //NEURALNETLIBRARY_NETFILEMANAGER_H
