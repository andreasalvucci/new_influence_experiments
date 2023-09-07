from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm
from parameterfree import COCOB
import torch
import numpy as np
import random
import json

import rdflib
from influence.data import Graph
from influence.models import UniformInfluencePredictor, FrequencyInfluencePredictor, InverseFrequencyInfluencePredictor, FitInfluencePredictor, DNNInfluencePredictor

parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, choices=["uniform", "frequency", "inverse-frequency", "fit", "dnn"], help="Model to use for the prediction.")
parser.add_argument("-i", "--input", type=Path, required=True, help="Input path to the RDF graph.")
parser.add_argument("-o", "--output", type=Path, required=True, help="Output path for the json file.")
parser.add_argument("-tp", "--target-pred", type=str, required=True, help="Target predicate to predict.")
parser.add_argument("-ip", "--ignore-predicate", action='append', help="Predicates to be ignored.")
parser.add_argument("-it", "--iterations", type=int, help="Iterations.", default=100)

parser.add_argument("-d", "--communicability-degree", type=int, default=5, help="Communicability degree used to compute f-communicability.")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for training.")
parser.add_argument("--test", type=float, default=0.2, help="Test size fraction.")

if __name__ == "__main__":
  args = parser.parse_args()

  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  torch.cuda.manual_seed(42)

  data = Graph(args.input)

  inf_pred = rdflib.URIRef(args.target_pred)
  ignore_preds = [rdflib.URIRef(p) for p in args.ignore_predicate]
  adj, target, train_idxs, test_idxs, considered_relations = data.torch(inf_pred, ignore_preds, device=args.device, test_size=0.3)

  if args.model == "uniform":
    m = UniformInfluencePredictor(len(considered_relations), target.shape[1], 
      comunicability_degree=args.communicability_degree, 
      device=args.device).to(args.device)
  elif args.model == "frequency":
    m = FrequencyInfluencePredictor(len(considered_relations), target.shape[1], 
      comunicability_degree=args.communicability_degree, 
      device=args.device).to(args.device)
  elif args.model == "inverse-frequency":
    m = InverseFrequencyInfluencePredictor(len(considered_relations), target.shape[1], 
      comunicability_degree=args.communicability_degree, 
      device=args.device).to(args.device)
  elif args.model == "fit":
    m = FitInfluencePredictor(len(considered_relations), target.shape[1], 
      comunicability_degree=args.communicability_degree, 
      device=args.device).to(args.device)
  elif args.model == "dnn":
    m = DNNInfluencePredictor(len(considered_relations), target.shape[1], 
      comunicability_degree=args.communicability_degree, 
      device=args.device).to(args.device)
  
  metrics = []
  train_losses = []
  test_losses = []  
  if m.TRAINABLE:
    optimizer = COCOB(m.parameters(), weight_decay=0.0)

    for step in (pbar := tqdm(list(range(args.iterations)))):
      loss, _ = m.forward(adj, target, train_idxs)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_losses.append(loss.item())

      if (step % 10) == 0:
        cur_m = {}
        loss_added = False
        for k in [None, 1, 5, 10, 20, 50, 100]:
          test_loss, k_metrics = m.evaluate(adj, target, test_idxs, k=k)

          if not loss_added:
            test_losses.append(test_loss.item())
            loss_added = True

          for n, v in k_metrics.items():
            cur_m[n] = v

        pbar.set_postfix_str(f"MRR = {cur_m['MRR@None']:1.5f} | MAP = {cur_m['MAP@None']:1.5f} | DCG = {cur_m['DCG@None']:1.5f}")
        metrics.append(cur_m)
      
      pbar.set_description(f"Train loss: {train_losses[-1]:1.8f} | Test loss {test_losses[-1]:1.8f}")
  else:
    m.forward(adj, target, test_idxs)
    
    loss_added = False
    metrics = {}
    for k in [None, 1, 5, 10, 20, 50, 100]:
      test_loss, k_metrics = m.evaluate(adj, target, test_idxs, k=k)

      if not loss_added:
        test_losses.append(test_loss.item())
        loss_added = True
      
      for k, v in k_metrics.items():
        metrics[k] = v
    
    for step in (pbar := tqdm([None])):
      pbar.set_postfix_str(f"MRR = {metrics['MRR@None']:1.5f} | MAP = {metrics['MAP@None']:1.5f} | DCG = {metrics['DCG@None']:1.5f}")

    metrics = [metrics]
    test_losses.append(test_loss.item())

  with open(args.output, "w") as f:
    json.dump({
      "relationship_weights": dict(zip(considered_relations, [x.item() for x in m.relation_weight])),
      "train_losses": train_losses,
      "test_losses": test_losses,
      "metrics": metrics
    }, f)
