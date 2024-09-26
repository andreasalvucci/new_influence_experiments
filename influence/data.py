from pathlib import Path
import networkx as nx
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict

class Graph:
  def __init__(self, src: Path):
    self.__src = src
    self.__rdf_graph = rdflib.Graph()
    self.__rdf_graph.parse(str(src))

    self.__preds = set(self.rdf.predicates())

  @property
  def rdf(self):
    return self.__rdf_graph
  
  @property
  def predicates(self):
    return self.__preds

  def torch(self, nodes, relations, target_pred, test_size: float = 0.2, device: str = "cpu"):
    # compute relations hyperplanes
    data = None
    for rel in relations:
      sub_g = nx.DiGraph()
      sub_g.add_nodes_from(nodes)
      sub_g.add_edges_from(self.rdf.subject_objects(rel))

      # 1 x N x N
      sub_g_adj = torch.tensor(nx.to_numpy_array(sub_g)).unsqueeze(0)

      # |R| x N x N
      data = sub_g_adj if data is None else torch.cat([data, sub_g_adj])

    # target data
    target_g = nx.DiGraph()
    target_g.add_nodes_from(nodes)
    target_g.add_edges_from(self.rdf.subject_objects(target_pred))

    # remove nodes
    target_g.remove_nodes_from(set(target_g).difference(nodes))

    target = torch.tensor(nx.to_numpy_array(target_g))

    # compute a mask for the nodes that have a target pred set
    influenced_mask = torch.where(target.sum(dim=1) > 0)[0]

    # split target pred set into train and test
    train_idxs, test_idxs = train_test_split(influenced_mask, test_size=test_size)

    data = data.to(device)
    target = target.to(device)
    train_idxs = train_idxs.to(device)
    test_idxs = test_idxs.to(device)

    return data, target, train_idxs, test_idxs, relations

  def sparse(self, nodes, relations, weight = dict()):
    d = defaultdict(lambda: defaultdict(float))

    node_idx = { node: idx for idx, node in enumerate(nodes) }
    for source, rel, target in tqdm(self.rdf):
      if rel in relations:
        source_idx = node_idx[source]
        target_idx = node_idx[target]
        d[source_idx][target_idx] += weight.get(rel, 0)

    rows = []
    cols = []
    vals = []
    max = None
    for row_idx, cols_d in d.items():
      for col_idx, weight in cols_d.items():
        rows.append(row_idx)
        cols.append(col_idx)
        vals.append(weight)

        if max is None or weight > max:
          max = weight
    
    data = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals) / max)
    return data