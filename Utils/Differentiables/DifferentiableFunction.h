//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_DIFFERENTIABLEFUNCTION_H
#define NEURALNETLIBRARY_DIFFERENTIABLEFUNCTION_H


struct DifferentiableFunction {
    virtual double func(double x) const = 0;
    virtual double derive(double x) const = 0;
};


#endif //NEURALNETLIBRARY_DIFFERENTIABLEFUNCTION_H
