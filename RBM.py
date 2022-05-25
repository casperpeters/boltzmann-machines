"""
This is a basic implementation of the restricted boltzmann machine as described in [1].
The code is created with much help of [2].
~ Casper Peters

References
----------
[1] : Salakhutdinov, R., Mnih, A., Hinton, G.E. (2007) Restricted Boltzmann Machines for Collaborative Filtering
[2] : Hinton G.E. (2012) A Practical Guide to Training Restricted Boltzmann Machines
"""

import torch
from tqdm import tqdm


class RBM(torch.nn.Module):
    def __init__(self, n_visible: int, n_hidden: int):
        """
        Initializes an RBM with random weights and zeros biases

        Parameters
        ----------
        n_visible : int
            The number of visible units of the RBM
        n_hidden : int
            The number of hidden units of the RBM
        """

        super(RBM, self).__init__()

        self.W = torch.nn.Parameter(torch.zeros(n_hidden, n_visible))
        self.b_v = torch.nn.Parameter(torch.zeros(1, n_visible))
        self.b_h = torch.nn.Parameter(torch.zeros(1, n_hidden))

        self.n_visible = n_visible
        self.n_hidden = n_hidden

    def visible_to_hidden(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the probabilities p(h=1|v) and a samples from this using a bernoulli distribution.

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

        p = torch.sigmoid(torch.matmul(v, self.W.T) + self.b_h)

        return p, torch.bernoulli(p)

    def hidden_to_visible(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the probabilities p(v=1|h) and a samples from this using a bernoulli distribution.

        Parameters
        ----------
        h : torch.Tensor
            The hidden layer activations.

        Returns
        -------
        p : torch.Tensor
            The visible conditional probabilities.
        v : torch.Tensor
            The visible layer samples.
        """

        p = torch.sigmoid(torch.matmul(h, self.W) + self.b_v)

        return p, torch.bernoulli(p)

    def contrastive_divergence(self, v: torch.Tensor, n: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs contrastive divergence (CD) n times.

        Parameters
        ----------
        v : torch.Tensor
            The visible layer of the RBM
        n : int
            The number of times to perform CD

        Returns
        -------
        v_n : torch.Tensor
            The visible layer after n iterations of CD
        h : torch.Tensor
            The initial hidden layer of the RBM
        h_n : torch.Tensor
            The hidden layer after n iterations of CD
        """

        if n < 1:
            raise ValueError('"n" must be greater than 0.')

        h, ph = self.visible_to_hidden(v)
        h_n = h.detach().clone()
        for _ in range(n):
            _, v_n = self.hidden_to_visible(h_n)
            ph_n, h_n = self.visible_to_hidden(v_n)

        return v_n, ph, ph_n

    def learn(self, data: torch.Tensor, n_epochs: int = 100, n: int = 5, lr: float = 0.01) -> torch.Tensor:
        """
        Trains an instance of an RBM

        Parameters
        ----------
        data : torch.Tensor of size (n_batches x n_visible)
            The data to train on.
        n_epochs : int, 100 by default
            The number of epochs to train for.
        n : int, 5 by default
            The number of Gibbs sampling steps to take per datapoint.
        lr : float, 0.001 by default
            The learning rate.

        Returns
        -------
        mean_square_errors : torch.Tensor
            The mean square errors at each epoch.
        """

        # empty tensor to save errors in
        mean_square_errors = torch.zeros(n_epochs)

        # loop over epochs
        for epoch in tqdm(range(n_epochs)):

            # loop over batches
            for batch in data:

                # add a dummy dimension to the batch
                batch = torch.unsqueeze(batch, dim=0)

                # perform contrastive divergence to compute model statistics
                v_n, h, h_n = self.contrastive_divergence(batch, n)

                # calculate gradient of model parameters using data and model statistics
                # tensor.squeeze removes excess dimension for clean calculations
                dW = torch.outer(h.squeeze(), batch.squeeze()) - torch.outer(h_n.squeeze(), v_n.squeeze())
                db_v = batch - v_n
                db_h = h - h_n

                # update weights and biases
                self.W += lr * dW
                self.b_v += lr * db_v
                self.b_h += lr * db_h

                # calculate errors
                mean_square_errors[epoch] += torch.mean((data - v_n) ** 2) / data.shape[0]

        return mean_square_errors

    def sample(self, v_start: torch.Tensor = None, n_samples: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function samples from the RBM using the Gibb's sampling method.

        Parameters
        ----------
        v_start : torch.Tensor
            The starting visible unit configuration.
        n_samples : int, 1000 by default
            The number of samples to generate.

        Returns
        -------
        vt : torch.Tensor
            The visible samples.
        ht : torch.Tensor
            The hidden samples.
        """

        if v_start is None:
            v_start = torch.randint(low=0, high=2, size=(self.n_visible, ), dtype=torch.float)

        # initialize empty tensors to save visible and hidden samples
        vt = torch.zeros(self.n_visible, n_samples, dtype=torch.float)
        ht = torch.zeros(self.n_hidden, n_samples, dtype=torch.float)

        v = v_start.unsqueeze(0)

        # loop over the number of samples
        for i in range(n_samples):
            _, h = self.visible_to_hidden(v)
            _, v = self.hidden_to_visible(h)

            vt[:, i] = v
            ht[:, i] = h

        return vt, ht
