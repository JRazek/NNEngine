//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "Optimizers/MBGD.h"
#include "Optimizers/MomentumGD.h"
#include <opencv2/opencv.hpp>
#include "Utils/Files/CSVReader.h"
#include "Utils/Files/ImageRepresentation.h"
#include <fstream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"

int main(){
    cn::Network network(9, 9, 1, 1);


    const int outputSize = 10;
    network.appendConvolutionLayer({3, 3},2, {1, 1}, {1, 1} );
    network.appendReLULayer();
    network.appendFlatteningLayer();
    network.appendFFLayer(outputSize);
    network.appendSigmoidLayer();
    network.initRandom();
    network.ready();

    cn::JSON json = network.jsonEncode();

    cn::MomentumGD momentumGd(network, 0.7, 0.01);


    CSVReader csvReader("/home/user/IdeaProjects/digitRecogniser/dataSet/metadata.csv", ';');
    csvReader.readContents();
    auto &contents = csvReader.getContents();
    std::vector<ImageRepresentation> imageRepresentations;
    imageRepresentations.reserve(contents.size());
    for(auto &c : contents){
        std::string path = c[0];
        std::string value = c[1];
        imageRepresentations.emplace_back(path, value);
    }


    cn::Bitmap<double> target (outputSize, 1, 1);
    for(int i = 0; i < outputSize; i ++){
        target.setCell(i, 0, 0, 0);
    }
    auto getBest = [](const cn::Bitmap<double> &output){
        int best = 0;
        for (int j = 0; j < output.w(); ++j) {
            if(output.getCell(j, 0, 0) > output.getCell(best, 0, 0)){
                best = j;
            }
        }
        return best;
    };

    std::shuffle(imageRepresentations.begin(), imageRepresentations.end(), std::default_random_engine(1));

    int resetRate = 100;
    int correctCount = 0;

    std::string filePath = "/home/user/networkBackup.json";

    constexpr int epochsCount = 100;
    for (u_int i = 0; i < imageRepresentations.size() * epochsCount; i++) {
        int n = i % imageRepresentations.size();
        ImageRepresentation &imageRepresentation = imageRepresentations[n];
        cv::Mat mat = cv::imread(imageRepresentation.path);

        cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
        bitmap = cn::Utils::average3Layers(bitmap);
        int numVal = std::stoi(imageRepresentation.value);
        target.setCell(numVal, 0, 0, 1);
        cn::Vector3<int> inputSize = network.getInputSize(0);
        network.feed(cn::Utils::resize(bitmap, inputSize.x, inputSize.y));
        momentumGd.propagate(target);

        int best = getBest(network.getNetworkOutput().value());
        if (best == numVal) {
            correctCount++;
        }

        if (!((i + 1) % resetRate)) {
            std::cout << "LOSS "<< i <<": "<< momentumGd.getError(target) << "\n";
            std::cout << "ACCURACY: " << (double) correctCount / double(resetRate) * 100 << "%\n";
            correctCount = 0;
        }

        target.setCell(numVal, 0, 0, 0);
        if((i + 1) % imageRepresentations.size()){
            //save time :)
            std::fstream file(filePath, std::ios::out);
            file << network.jsonEncode();
            file.close();
        }
    }
    return 0;
}

/*
 *
 *          cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
            bitmap = cn::Utils::resize(bitmap, 100, 100);
            bitmap = cn::Utils::average3Layers(bitmap);
            cn::byte *tmp = new cn::byte[bitmap.size().multiplyContent()];
            cn::Utils::convert(bitmap.data(), tmp, bitmap.w(), bitmap.h(), bitmap.d(), 0, 1);

            cv::Mat resized (bitmap.w(), bitmap.h(), CV_8UC(bitmap.d()), tmp);


            cv::imshow("", resized);
            cv::waitKey(10000);
            delete[] tmp;

            return 0;
 * */