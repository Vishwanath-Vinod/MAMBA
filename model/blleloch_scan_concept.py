import torch
import math
import torch.nn.functional as F

"""
This is the parallelized Blleloch scan used in MAMBA recurrence. This code is for clarity in algorithm alone.
It is memory inefficient as it does not use in-place tensors but rather conveys the idea better using binary trees.
For the efficient, more complex parallel scan used refer parallel_scan.py.
"""
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.prefix = None

class ParallelScan:
    def __init__(self,A,B):
      # A and Bu passed as arguments are (B,D,L,N) shape
      A_seq  = A.permute(2,0,1,3)   # (L,B,D,N)
      Bu_seq = B.permute(2,0,1,3)  # (L,B,D,N)
      self.A = A_seq
      self.B = Bu_seq

    def composition(self,node_left,node_right):
      "Shows how composition works for MAMBA recurrence operation"
      Al,Bl = node_left
      Ar,Br = node_right
      A_compose = Ar*Al
      B_compose = Ar*Bl+Br
      return (A_compose,B_compose)

    def upsweep(self,nodes):
      "Generates the binary tree obtained as a result of the upsweep."
      if len(nodes) == 1:
        return nodes[0]

      parents = []

      for i in range(0, len(nodes), 2):
          val = self.composition(nodes[i].value,nodes[i+1].value)
          parents.append( Node(val,left=nodes[i],right=nodes[i+1]))

      return self.upsweep(parents)

    def downsweep(self, node, prefix):
      """
      node   : current tree node
      prefix : reduction of everything before this subtree
      """
      if node.left is None and node.right is None:
          node.prefix = prefix
          return

      left_prefix = prefix
      right_prefix = self.composition(prefix,node.left.value)

      self.downsweep(node.left, left_prefix)
      self.downsweep(node.right, right_prefix)

    def scan(self):
      """
      Implements the Blleloch scan and obtains as a list of prefixes the operations to be performed on each token.
      prefixes[t] returns (A_prefix_t, B_prefix_t) representing the composition of everything before timestep t
      states is then found by composing this prefix on each operation and then since h_(-1)=0. This composition is the state.
      """
      nodes = [Node((a,b)) for a,b in zip(self.A,self.B)]
      L = self.A.shape[0]
      root = self.upsweep(nodes)
      identity = (torch.ones_like(self.A[0]),torch.zeros_like(self.B[0]),)
      identity = (1,0)
      self.downsweep(root, identity)
      prefixes = [node.prefix for node in nodes]
      states = torch.stack([self.composition(prefixes[t], (self.A[t], self.B[t]))[1]for t in range(L)],dim=0) #(L,B,D,N)
      states = states.permute(1,2,0,3)  # (B,D,L,N)
      return states