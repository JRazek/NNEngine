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
        virtual JSON jsonEncode() const = 0;
    };
    template<typename T>
    class Tensor;
}
#endif //NEURALNETLIBRARY_JSONENCODABLE_H
