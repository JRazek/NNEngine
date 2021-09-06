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
#include "../Utils/interfaces/JSONEncodable.h"
#include <memory>

namespace cn {

    class Network : public JSONEncodable{
    private:
        int seed;
        /**
         * what the dimensions of the byte array is after being normalized and sampled
         */
        Vector3<int> inputSize;
        std::default_random_engine randomEngine;
        std::vector<std::unique_ptr<Layer>> allocated;

    protected:
        std::vector<Learnable *> learnableLayers;
        std::vector<Layer *> layers;

        std::optional<Bitmap<double>> input;
        std::vector<Bitmap<double>> outputs;

        std::optional<OutputLayer> outputLayer;

    public:

        void appendConvolutionLayer(int kernelX, int kernelY, int kernelsCount, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0);
        void appendFFLayer(int neuronsCount);
        void appendFlatteningLayer();
        void appendBatchNormalizationLayer();
        void appendMaxPoolingLayer(int kernelSizeX, int kernelSizeY);
        void appendReluLayer();
        void appendSigmoidLayer();




        /**
         *
         * @param takes _input in 1 format type
         */
        void feed(const byte *_input);

        void feed(Bitmap<double> bitmap);
        void feed(const Bitmap<cn::byte> &bitmap);

        /**
         * when network structure is ready - run this function.
         */
        void ready();


        /**
         *
         * @return learnables
         */
        const std::vector<Learnable *> &getLearnables() const;


        const std::vector<Layer *> &getLayers() const;


        /**
         * input size for feed
         * @param w width
         * @param h height
         * @param d depth
         * @param _seed - seed for random engine
         * if the first layer is image - set all the properties.
         * In case of using only FFLayers - set height and depth to 1.
         */

        Network(int w, int h, int d, int _seed = 1);


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
         * @param low lower bound for number
         * @param high higher bound for number
         * @return pseudorandom number with seed given in constructor
         */
        double getRandom(double low, double high);


        /**
         *
         * @param layerID
         * @return the size vector of nth layer's output
         */
        Vector3<int> getOutputSize(int layerID) const;

        /**
         *
         * @param layerID
         * @return returns input size for each layer
         */
        Vector3<int> getInputSize(int layerID) const;

        /**
         *
         * @param layerID layer we want to get the chain from
         * @param inputPos
         * @return chain from requested layer
         */
        double getChain(int layerID, const Vector3<int> &inputPos);

        void resetMemoization();

        const Bitmap<double> &getInput(int layerID) const;
        const Bitmap<double> &getNetworkOutput() const;
        const Bitmap<double> &getOutput(int layerID) const;

        JSON jsonEncode() const override;

        OutputLayer &getOutputLayer();
    };
}


#endif //NEURALNETLIBRARY_NETWORK_H
