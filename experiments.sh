#!/bin/bash

for i in {1..5}
do
  for exp in {1..5}
  do
    for model in fit dnn
    do
      python main.py \
        -m $model \
        -i resources/linked_jazz_legacy.ttl \
        -tp "http://w3id.org/friendshipneverends/ontology/admires" \
        -ip "http://w3id.org/friendshipneverends/ontology/hasAdmirator" \
        --device "cuda:0" \
        -d $i \
        -it 300 \
        -o "experiments/$model.$i.$exp.json"
    done
  done
done


  