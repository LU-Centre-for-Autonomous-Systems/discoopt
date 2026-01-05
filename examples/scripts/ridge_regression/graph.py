import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from topolink import Graph

L = np.array([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]])
W = np.eye(4) - L * 0.2

graph = Graph.from_mixing_matrix(W)

graph.deploy()
