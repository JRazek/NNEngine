//
// Created by jrazek on 06.09.2021.
//

#ifndef NEURALNETLIBRARY_JSONENCODABLE_H
#define NEURALNETLIBRARY_JSONENCODABLE_H
#include <string>
#include <nlohmann/json.hpp>

namespace cn{
    using JSON = nlohmann::json;
    struct JSONEncodable {
        virtual JSON jsonEncode() = 0;
    };
    template<typename T>
    class Bitmap;
}
namespace ns{
    template<typename T>
    void to_json(cn::JSON &json, const T t){

    }
}
#endif //NEURALNETLIBRARY_JSONENCODABLE_H
