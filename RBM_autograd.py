"""
This is an implementation of the restricted boltzmann machine as described in [1] using torch autograd function.
The code is created with much help of [2] and the torch documentation.
~ Casper Peters

References
----------
[1] : Salakhutdinov, R., Mnih, A., Hinton, G.E. (2007) Restricted Boltzmann Machines for Collaborative Filtering
[2] : Hinton G.E. (2012) A Practical Guide to Training Restricted Boltzmann Machines
"""

import torch
from tqdm import tqdm

from RBM import RBM as RBM_


class RBM(RBM_):
    def __init__(self, n_visible: int, n_hidden: int):
        """
        Initializes an RBM with random weights and zeros biases

        Parameters
        ----------
        n_visible : int
            The number of visible units of the RBM
        n_hidden : int
            The number of hidden units of the RBM
        dtype : torch.dtype, optional
            Data type of the tensor.
        """

        super(RBM, self).__init__(n_visible, n_hidden)

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the free energy of an RBM given a visible vector.

        Parameters
        ----------
        v : torch.Tensor
            Visible vector.

        Returns
        -------
        torch.Tensor
            Free energy of the visible vector.
        """

        F = -torch.inner(v, self.b_v) - torch.sum(torch.log(1 + torch.exp(torch.matmul(v, self.W.T) + self.b_h)), dim=1)

        return F

    def learn(self, data, n_epochs=100, n=5, lr: float = 0.01):
        """
        Trains an instance of an RBM

        Parameters
        ----------
        data : torch.Tensor of size (n_batches x n_visible)
            The data to train on.
        n_epochs : int
            The number of epochs to train for.
        n : int
            The number of Gibbs sampling steps to take per datapoint.
        lr : float
            The learning rate.

        Returns
        -------
        mean_square_errors : torch.Tensor
            The mean square errors at each epoch.
        """

        # initialize the optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # empty tensor to save loss in
        losses = torch.zeros(n_epochs)

        # loop over epochs
        for epoch in tqdm(range(n_epochs)):

            # loop over batches
            for batch in data:

                # add a dummy dimension to the batch
                batch = torch.unsqueeze(batch, dim=0)

                # perform contrastive divergence to compute model statistics
                v_n, h, h_n = self.contrastive_divergence(batch, n)

                # calculate loss
                loss = torch.mean(self.free_energy(batch) - self.free_energy(v_n.detach()))

                # calculate gradients and update parameters with torch autograd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # save loss
                losses[epoch] += loss.clone().detach() / data.shape[0]

        return losses
