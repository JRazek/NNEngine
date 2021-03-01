#pragma once
struct ActivationFunction{
    ActivationFunction(){};
    virtual float operator()(float x) const = 0;
    float getValue(float x){
        return  (*this)(x);
    }
};