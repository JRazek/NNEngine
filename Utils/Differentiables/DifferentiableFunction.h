//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_DIFFERENTIABLEFUNCTION_H
#define NEURALNETLIBRARY_DIFFERENTIABLEFUNCTION_H


struct DifferentiableFunction {
    virtual float func(float x) const = 0;
    virtual float derive(float x) const = 0;
};


#endif //NEURALNETLIBRARY_DIFFERENTIABLEFUNCTION_H
