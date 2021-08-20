//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include <vector>
#include <random>
#include "../Utils/Utils.h"
#include "layers/interfaces/RandomInitiable.h"

namespace cn {
    class Layer;
    class FFLayer;
    class ConvolutionLayer;

    class Network {
        std::default_random_engine randomEngine;

    protected:
        std::vector<RandomInitiable *> randomInitLayers;
        std::vector<Layer *> layers;
        friend class Layer;

        friend ConvolutionLayer;

        void appendLayer(Layer * layer);

    public:

        void appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount, const DifferentiableFunction &differentiableFunction, int paddingX = 0,
                                    int paddingY = 0, int strideX = 1, int strideY = 1);

        void appendFFLayer(int neuronsCount, const DifferentiableFunction &differentiableFunction);

        void appendFlatteningLayer();


        /**
         * what the dimensions of the byte array is after being normalized and sampled
         */

        const int inputDataWidth;
        const int inputDataHeight;
        const int inputDataDepth;


        /**
         *
         * @param takes input in 1 format type
         */
        void feed(const byte *input);

        void feed(const Bitmap<float> &bitmap);
        void feed(const Bitmap<cn::byte> &bitmap);


        /**
         *
         * @return layers
         */
        const std::vector<Layer *> * getLayers();


        /**
         * input size for feed
         * @param w width
         * @param h height
         * @param d depth
         * @param seed - seed for random engine
         * if the first layer is image - set all the properties.
         * In case of using only FFLayers - set height and depth to 1.
         */

        Network(int w, int h, int d, int seed = 0);


        /**
         * not supported yet
         */
        Network(const Network&) = delete;

        void initRandom();

        float getRandom(float low, float high);

        ~Network();
    };
}


#endif //NEURALNETLIBRARY_NETWORK_H
