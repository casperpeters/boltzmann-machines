"""
This is an implementation of the restricted boltzmann machine with hidden Rectified Linear Unit potential [1].
~ Casper Peters

References
----------
[1] : Nair, V., & Hinton, G. E. (2010, January). Rectified linear units improve restricted boltzmann machines. In Icml.
"""

import torch
import torch.nn.functional as F
from RBM_autograd import RBM


class ReLuRBM(RBM):
    def __init__(self, n_visible, n_hidden):
        """
        Initializes an RBM with ReLu hidden unit potential

        Parameters
        ----------
        n_visible : int
            The number of visible units of the RBM
        n_hidden : int
            The number of hidden units of the RBM
        """

        super(ReLuRBM, self).__init__(n_visible, n_hidden)

    def visible_to_hidden(self, v):
        """
        Computes the probabilities p(h=1|v) and samples for the hidden unit activation layer using a ReLu potential

        Parameters
        ----------
        v : torch.Tensor
            The visible layer of the RBM.

        Returns
        -------
        p : torch.Tensor
            The hidden conditional probabilities.
        h : torch.Tensor
            The hidden layer samples.
        """

        activations = F.linear(v, self.W) + self.b_h
        states = F.relu(activations)
        probs = states

        return probs, states
