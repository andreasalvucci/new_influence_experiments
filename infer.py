from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm
from parameterfree import COCOB
import torch
import numpy as np
import random
import json
from math import factorial
from tqdm import tqdm

import rdflib
from influence.data import Graph
from influence.models import UniformInfluencePredictor, FrequencyInfluencePredictor, InverseFrequencyInfluencePredictor, FitInfluencePredictor, DNNInfluencePredictor

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, required=True, help="Input path to the RDF graph.")
parser.add_argument("-w", "--weights", type=Path, required=True, help="Path to the JSON file containing ")
parser.add_argument("-o", "--output", type=Path, required=True, help="Output path to the enriched RDF graph.")
parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for the influence detection.")
parser.add_argument("-d", "--communicability-degree", type=int, default=5, help="Communicability degree used to compute f-communicability.")

if __name__ == "__main__":
  args = parser.parse_args()

  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  torch.cuda.manual_seed(42)

  relationship = json.load(open(args.weights))
  relationship = {
    rdflib.URIRef(k): v
    for k, v in relationship.items()
  }

  graph = Graph(args.input)

  # collect all the nodes connected to the relation
  nodes = set()
  for rel in relationship.keys():
    # add all the node as rel subjects
    for subj, _, obj in graph.rdf.triples((None, rel, None)):
      nodes.add(subj)
      nodes.add(obj)

  nodes = list(nodes)
  
  data = graph.sparse(list(nodes), list(relationship.keys()), weight=relationship)
  print("Adjacency computed")
  
  initial_graph = data
  for i in range(2, args.communicability_degree + 1):
    coeff = torch.tensor(1 / factorial(i))
    data = data + (coeff * torch.sparse.mm(data, initial_graph))

  print("f-communicability computed")

  data = data.coalesce()
  # influence mask
  mask = data.indices()[:, data.values() > args.threshold]

  influece_p = rdflib.URIRef("https://w3id.org/polifonia/ontology/relationship/admires")
  for mask_idx in tqdm(list(range(mask.shape[1]))):
    source = nodes[mask[0, mask_idx]]
    target = nodes[mask[1, mask_idx]]

    if source != target:
      graph.rdf.add((source, influece_p, target))

  # export the enriched graph
  graph.rdf.serialize(args.output)