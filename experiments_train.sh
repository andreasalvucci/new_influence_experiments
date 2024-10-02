#!/bin/bash

for i in {1..5}
do
  for exp in {1..5}
  do
    for model in fit #dnn  uniform frequency inverse-frequency
    do
      python train.py \
        -m $model \
        -i resources/linked_jazz.nt \
        -tp "https://w3id.org/polifonia/ontology/relationship/admires" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasRival" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasAcquaintance" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasBandmate" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasChild" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasCopupil" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasFellow" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasFriend" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasMentor" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasParent" \
        -ip "https://w3id.org/polifonia/ontology/relationship/hasPupil" \
        --device "cuda:0" \
        -d $i \
        -it 1000 \
        -o "experiments/training/$model.$i.$exp.json"
    done
  done
done


  