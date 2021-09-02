#!/bin/bash

dataSetsPath=/home/jrazek/IdeaProjects/digitRecogniser/dataSet
csvOutputFile=/home/jrazek/IdeaProjects/digitRecogniser/dataSet/metadata.csv

touch metadata.csv
cat /dev/null > $csvOutputFile

for i in {0..9}; do
  for filename in $dataSetsPath/$i/*.jpg; do
   name=$(basename $filename)
   echo "$filename;$i" >> $csvOutputFile
  done
done