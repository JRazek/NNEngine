//
// Created by jrazek on 02.09.2021.
//

#ifndef NEURALNETLIBRARY_NETFILEMANAGER_H
#define NEURALNETLIBRARY_NETFILEMANAGER_H

#include <nlohmann/json.hpp>
#include "../../Network/Network.h"
namespace cn{
    using json = nlohmann::json;

    class NetFileManager {
        void saveNet(const Network &network){

        }

        void test() {
            auto jsonObject = json::parse("{\"array\":[1,2,3,4,5,6,7,8,9,10]}");

        }
    };
}

#endif //NEURALNETLIBRARY_NETFILEMANAGER_H
