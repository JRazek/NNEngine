//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "LearningModels/Backpropagation.h"
#include <opencv2/opencv.hpp>
#include "Utils/Files/CSVReader.h"
#include "Utils/Files/ImageRepresentation.h"

int main(){
    cn::Network network(100, 100, 1, 2);

    ReLU reLu;
    Sigmoid sigmoid;

    cn::Backpropagation backpropagation(network, 0.1);

    const int outputSize = 10;
    network.appendConvolutionLayer(3, 3, 2, reLu, 0, 0, 1, 1);
    network.appendMaxPoolingLayer(4, 4);
    network.appendBatchNormalizationLayer();
    network.appendConvolutionLayer(3, 3, 10, reLu, 0, 0, 2, 2);
    network.appendMaxPoolingLayer(2, 2);
    network.appendConvolutionLayer(3, 3, 20, reLu, 0, 0, 1, 1);
    network.appendBatchNormalizationLayer();
    network.appendFlatteningLayer();
    network.appendFFLayer(20, sigmoid);
    network.appendFFLayer(outputSize, sigmoid);
    network.initRandom();
    network.ready();





    CSVReader csvReader("/home/user/CLionProjects/dataSets/training-b.csv", ',');
    csvReader.readContents();
    auto &contents = csvReader.getContents();
    std::vector<ImageRepresentation> imageRepresentations;
    imageRepresentations.reserve(contents.size());
    for(auto &c : contents){
        std::string path = c[6] + "/" + c[0];
        std::string value = c[3];
        imageRepresentations.emplace_back(path, value);
    }

    cn::Bitmap<float> target (outputSize, 1, 1);
    for(int i = 0; i < outputSize; i ++){
        target.setCell(i, 0, 0, 0);
    }

    for(int i = 0; i < imageRepresentations.size(); i ++) {
        cv::Mat mat = cv::imread(imageRepresentations[i].path);
        cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
        bitmap = cn::Utils::resize(bitmap, 100, 100);
        network.feed(bitmap);
        int numVal = std::stoi(imageRepresentations[i].value);
        target.setCell(numVal - 1, 0, 0, 1);
        std::cout<<i<<": "<<backpropagation.getError(target)<<"\n";
        backpropagation.propagate(target);
        target.setCell(numVal - 1, 0, 0, 0);
    }

    //PrefixSum2D<long long> prefixSum2D(bitmap);
    return 0;
}