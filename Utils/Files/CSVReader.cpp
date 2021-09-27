//
// Created by user on 29.08.2021.
//

#include "CSVReader.h"

CSVReader::CSVReader(const std::string &_filePath, char _splitter):filepath(_filePath), file(_filePath, std::ios::in), splitter(_splitter){}

void CSVReader::readContents() {
    if(file.is_open()){
        int lineNum = 0;
        std::string line;
        while(getline(file, line)){
            contents.push_back(split(line, splitter));
            lineNum ++;
        }
        file.close();
    }
}

std::vector<std::string> CSVReader::split(const std::string &str, char splitter) {
    std::vector<std::string> result;
    std::string buff = "";
    for(u_int i = 0; i < str.size(); i ++){
        if(str[i] == splitter){
            result.push_back(buff);
            buff = "";
        }else{
            buff += str[i];
        }
    }
    if(buff.size())
        result.push_back(buff);
    return result;
}

const std::vector<std::vector<std::string>> &CSVReader::getContents() const{
    return contents;
}

