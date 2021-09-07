//
// Created by jrazek on 07.09.2021.
//

#ifndef NEURALNETLIBRARY_NETFILEMANAGER_H
#define NEURALNETLIBRARY_NETFILEMANAGER_H
#include "../../Network/layers/interfaces/Layer.h"
#include "../../Network/layers/FFLayer.h"
#include "../../Network/Network.h"
#include <memory>
namespace cn {
    class NetFileManager {
        static cn::Network networkFromJSON(const JSON &json){
            Network network(json["input_size"], json["seed"]);
            for(JSON layer : json["layers"]){

            }
            return network;
        }
    };
}

#endif //NEURALNETLIBRARY_NETFILEMANAGER_H
