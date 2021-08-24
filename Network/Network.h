//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include <vector>
#include <random>
#include "../Utils/Utils.h"
#include "layers/interfaces/Learnable.h"
#include "layers/OutputLayer.h"

namespace cn {
    class OutputLayer;

    class Network {
        std::default_random_engine randomEngine;

    protected:
        std::vector<Learnable *> randomInitLayers;
        std::vector<Layer *> layers;
        std::optional<Bitmap<float>> input;
        //OutputLayer outputLayer;
        friend class OutputLayer;

    public:

        void appendConvolutionLayer(int kernelX, int kernelY, int kernelsCount, const DifferentiableFunction &differentiableFunction, int paddingX = 0,
                                    int paddingY = 0, int strideX = 1, int strideY = 1);

        void appendFFLayer(int neuronsCount, const DifferentiableFunction &differentiableFunction);

        void appendFlatteningLayer();

        void appendBatchNormalizationLayer();

        void appendMaxPoolingLayer(int kernelSizeX, int kernelSizeY);


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
        const std::vector<Layer *> *getLayers();


        /**
         * input size for feed
         * @param w width
         * @param h height
         * @param d depth
         * @param seed - seed for random engine
         * if the first layer is image - set all the properties.
         * In case of using only FFLayers - set height and depth to 1.
         */

        Network(int w, int h, int d, int seed = 1);


        /**
         * not supported yet
         */
        Network(const Network&) = delete;

        /**
         * randomly initialized all the weights and biases of the network
         */
        void initRandom();


        /**
         *
         * @return the output from the last layer of the network
         */
        Bitmap<float> & getOutput();

        /**
         *
         * @param low lower bound for number
         * @param high higher bound for number
         * @return pseudorandom number with seed given in constructor
         */
        float getRandom(float low, float high);

        ~Network();
    };
}


#endif //NEURALNETLIBRARY_NETWORK_H
