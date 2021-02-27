#pragma once
struct ActivationFunction{
    virtual float operator()(float x) const;
};