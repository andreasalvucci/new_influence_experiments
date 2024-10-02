#!/bin/bash

#for i in {1..1}
#do
for exp in {1..1}
do
  for model in fit #dnn # uniform frequency inverse-frequency
  do
    python train.py \
      -m $model \
      -i resources/all_relations.ttl \
      -tp "http://w3id.org/friendshipneverends/ontology/hasAdmirator" \
      -ip "http://w3id.org/friendshipneverends/ontology/isFriendOf" \
      -ip "http://w3id.org/friendshipneverends/ontology/sameRecordLabel" \
      -ip "http://w3id.org/friendshipneverends/ontology/hasFriend" \
      -ip "http://dbpedia.org/ontology/sameGenre" \
      -ip "http://w3id.org/friendshipneverends/ontology/hasMentor" \
      -ip "http://w3id.org/friendshipneverends/ontology/hasBandmate" \
      -ip "http://w3id.org/friendshipneverends/ontology/hasSameMentor" \
      -ip "http://w3id.org/friendshipneverends/ontology/hasPupil" \
      -ip "http://w3id.org/friendshipneverends/ontology/sameBirthPlace" \
      -ip "http://w3id.org/friendshipneverends/ontology/hasAcquaintance" \
      -ip "http://w3id.org/friendshipneverends/ontology/teensInSameTimeandPlace" \
      --device "cuda:0" \
      -d 3 \
      -it 300 \
      -o "experiments/training/$model.3.$exp.json"
  done
done
#done