#!/bin/bash

dataSetsPath=/home/jrazek/CLionProjects/dataSets
finalPath=/home/jrazek/CLionProjects/dataSets/dataSet
csvOutputFile=metadata.csv

mkdir -p finalPath
touch metadata.csv
cat /dev/null > $csvOutputFile

for i in {0..9}; do
  for filename in $dataSetsPath/$i/*.jpg; do
   name=$(basename $filename)
   #cp $filename $finalPath/$i/$name
   echo "$filename;$i" >> $csvOutputFile
  done
done