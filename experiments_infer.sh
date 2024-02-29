#!/bin/bash

for i in `LC_NUMERIC=C; seq .1 .05 1` 
do
  python infer.py -i resources/meetups.ttl \
                  -w experiments/inference/relationships_weight.json \
                  -d 2 \
                  -o experiments/inference/$i.ttl \
                  -t $i
done


  