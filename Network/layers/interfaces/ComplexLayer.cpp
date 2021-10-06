//
// Created by user on 06.10.2021.
//

#include "ComplexLayer.h"

cn::ComplexLayer::ComplexLayer(cn::Vector3<int> _inputSize) : Layer(_inputSize) {}

cn::ComplexLayer::ComplexLayer(const cn::Layer &layer) : Layer(layer) {}

cn::ComplexLayer::ComplexLayer(cn::Layer &&layer) : Layer(layer) {}
