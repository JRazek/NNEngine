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
    cn::Network network(18, 18, 3, 243);

    ReLU reLu;
    Sigmoid sigmoid;

    cn::Backpropagation backpropagation(network, 1);

    const int outputSize = 10;
    network.appendConvolutionLayer(3, 3, 2, reLu, 0, 0, 1, 1);
    network.appendBatchNormalizationLayer();
    network.appendConvolutionLayer(3, 3, 10, reLu, 0, 0, 1, 1);
    network.appendBatchNormalizationLayer();
    network.appendFlatteningLayer();
    network.appendFFLayer(10, sigmoid);
    network.appendFFLayer(10, sigmoid);
    network.appendFFLayer(outputSize, sigmoid);
    network.initRandom();
    network.ready();





    CSVReader csvReader("/home/user/CLionProjects/dataSets/metadata.csv", ';');
    csvReader.readContents();
    auto &contents = csvReader.getContents();
    std::vector<ImageRepresentation> imageRepresentations;
    imageRepresentations.reserve(contents.size());
    for(auto &c : contents){
        std::string path = c[0];
        std::string value = c[1];
        imageRepresentations.emplace_back(path, value);
    }

    cn::Bitmap<float> target (outputSize, 1, 1);
    for(int i = 0; i < outputSize; i ++){
        target.setCell(i, 0, 0, 0);
    }
    auto getBest = [](const cn::Bitmap<float> &output){
        int best = 0;
        for (int j = 0; j < output.w(); ++j) {
            if(output.getCell(j, 0, 0) > output.getCell(best, 0, 0)){
                best = j;
            }
        }
        return best;
    };

    std::shuffle(imageRepresentations.begin(), imageRepresentations.end(), std::default_random_engine(1));

    int correctCount = 0;
    int resetRate = 100;
    for(int i = 0; i < imageRepresentations.size(); i ++) {
        ImageRepresentation &imageRepresentation = imageRepresentations[i];
        cv::Mat mat = cv::imread(imageRepresentation.path);
        cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
        network.feed(bitmap);
        int numVal = std::stoi(imageRepresentation.value);
        target.setCell(numVal, 0, 0, 1);
        std::cout<<i<<": "<<backpropagation.getError(target)<<"\n";
        backpropagation.propagate(target);
        int best = getBest(network.getNetworkOutput());
        if(best == numVal){
            correctCount ++;
        }

        if(!((i + 1) % resetRate)){
            std::cout<<"ACCURACY: "<<(float)correctCount / float(resetRate) * 100<<"%\n";
            correctCount = 0;
        }

        target.setCell(numVal, 0, 0, 0);
    }

    //PrefixSum2D<long long> prefixSum2D(bitmap);
    return 0;
}