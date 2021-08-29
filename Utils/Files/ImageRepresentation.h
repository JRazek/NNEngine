//
// Created by user on 29.08.2021.
//

#ifndef NEURALNETLIBRARY_IMAGEREPRESENTATION_H
#define NEURALNETLIBRARY_IMAGEREPRESENTATION_H
#include <string>

struct ImageRepresentation {
    ImageRepresentation(const std::string &_path, const std::string &_value);
    std::string path;
    std::string value;
};


#endif //NEURALNETLIBRARY_IMAGEREPRESENTATION_H
