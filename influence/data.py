from pathlib import Path
import networkx as nx
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
import torch
from sklearn.model_selection import train_test_split

class Graph:
  def __init__(self, src: Path):
    self.__src = src
    self.__rdf_graph = rdflib.Graph()
    self.__rdf_graph.parse(str(src))
    self.__nx_graph = rdflib_to_networkx_digraph(self.__rdf_graph)

    self.__preds = set(self.rdf.predicates())

  @property
  def rdf(self):
    return self.__rdf_graph
  
  @property
  def nx(self):
    return self.__nx_graph

  @property
  def predicates(self):
    return self.__preds

  def torch(self, target_pred, ignore_preds, test_size: float = 0.2, device: str = "cpu"):
    to_ignore = set(ignore_preds).union([target_pred])

    # prepare data
    relations = list(self.predicates - to_ignore)
    
    # compute relations hyperplanes
    data = None
    for rel in relations:
      sub_g = nx.DiGraph()
      sub_g.add_nodes_from(self.nx.nodes)
      sub_g.add_edges_from(self.rdf.subject_objects(rel))

      # 1 x N x N
      sub_g_adj = torch.tensor(nx.to_numpy_array(sub_g)).unsqueeze(0)

      # |R| x N x N
      data = sub_g_adj if data is None else torch.cat([data, sub_g_adj])

    # target data
    target_g = nx.DiGraph()
    target_g.add_nodes_from(self.nx.nodes)
    target_g.add_edges_from(self.rdf.subject_objects(target_pred))
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



  