# Copyright (c) OpenMMLab. All rights reserved.
from math import pi

import numpy as np
import torch


class GaussianMixture():
    """Initializes the Gaussian mixture model and brings all tensors into their
    required shape.

    Args:
        n_components (int): number of components.
        n_features (int, optional): number of features.
        mu_init (torch.Tensor, optional): (T, k, d)
        var_init (torch.Tensor, optional): (T, k, d) or (T, k, d, d)
        eps (float, optional): Defaults to 1e-6.
        requires_grad (bool, optional): Defaults to False.
    """

    def __init__(self,
                 n_components,
                 n_features=2,
                 mu_init=None,
                 var_init=None,
                 eps=1.e-6,
                 requires_grad=False):
        self.n_components = n_components
        self.n_features = n_features
        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps
        self.lower_bound_logdet = -783.
        self.requires_grad = requires_grad
        self.T = 1
        self.N = 9

    def _init_params(self, mu_init=None, var_init=None):
        """Initializes the parameters of Gaussian mixture model.

        Args:
            mu_init (torch.Tensor, optional): mu of Gaussian.
            var_init (torch.Tensor, optional): variance of Gaussian.
        """
        self.log_likelihood = -np.inf

        if mu_init is not None:
            self.mu_init = mu_init
        if var_init is not None:
            self.var_init = var_init

        if self.requires_grad:
            if self.mu_init is not None:
                assert torch.is_tensor(self.mu_init)
                assert self.mu_init.size() == (
                    self.T, self.n_components, self.n_features
                ), 'Input mu_init does not have required tensor dimensions' \
                   ' (%i, %i, %i)' % (
                       self.T, self.n_components, self.n_features)
                self.mu = self.mu_init.clone().requires_grad_().cuda()
            else:
                self.mu = torch.randn(
                    (self.T, self.n_components, self.n_features),
                    requires_grad=True).cuda()

            if self.var_init is not None:
                assert torch.is_tensor(self.var_init)
                assert self.var_init.size() == (
                    self.T, self.n_components, self.n_features,
                    self.n_features), 'Input var_init does not have required' \
                                      ' tensor dimensions (%i, %i, %i, %i)' % \
                                      (self.T, self.n_components,
                                       self.n_features,
                                       self.n_features)
                self.var = self.var_init.clone().requires_grad_().cuda()
            else:
                self.var = torch.eye(self.n_features).reshape(
                    (1, 1, self.n_features, self.n_features))\
                    .repeat(self.T, self.n_components, 1, 1)\
                    .requires_grad_().cuda()

            self.pi = torch.empty(
                (self.T, self.n_components,
                 1)).fill_(1. / self.n_components).requires_grad_().cuda()
        else:
            if self.mu_init is not None:
                assert torch.is_tensor(self.mu_init)
                assert self.mu_init.size() == (
                    self.T, self.n_components, self.n_features
                ), 'Input mu_init does not have required tensor dimensions' \
                   ' (%i, %i, %i)' % (
                       self.T, self.n_components, self.n_features)
                self.mu = self.mu_init.clone().cuda()
            else:
                self.mu = torch.randn(
                    (self.T, self.n_components, self.n_features)).cuda()

            if self.var_init is not None:
                assert torch.is_tensor(self.var_init)
                assert self.var_init.size() == (
                    self.T, self.n_components, self.n_features,
                    self.n_features), 'Input var_init does not have required' \
                                      ' tensor dimensions (%i, %i, %i, %i)' % \
                                      (self.T, self.n_components,
                                       self.n_features,
                                       self.n_features)
                self.var = self.var_init.clone().cuda()
            else:
                self.var = torch.eye(self.n_features).reshape(
                    (1, 1, self.n_features,
                     self.n_features)).repeat(self.T, self.n_components, 1,
                                              1).cuda()

            self.pi = torch.empty((self.T, self.n_components,
                                   1)).fill_(1. / self.n_components).cuda()

        self.params_fitted = False

    def check_size(self, x):
        """Make sure that the shape of x is (T, n, 1, d).

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(2)

        return x

    def fit(self, x, delta=1e-3, n_iter=10):
        """Fits Gaussian mixture model to the data.

        Args:
            x (torch.Tensor): input tensor.
            delta (float, optional): threshold.
            n_iter (int, optional): number of iterations.
        """
        self.T = x.size()[0]
        self.N = x.size()[1]

        select = torch.randint(self.N, size=(self.T * self.n_components, ))
        mu_init = x.reshape(-1, self.n_features)[select, :].view(
            self.T, self.n_components, self.n_features)
        self._init_params(mu_init=mu_init)

        x = self.check_size(x)
        i = 0
        j = np.inf

        while (i <= n_iter) and (not torch.is_tensor(j) or (j >= delta).any()):
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.em_runner(x)
            self.log_likelihood = self.get_score(x)

            if (self.log_likelihood.abs() == float('Inf')).any() or \
                    (torch.isnan(self.log_likelihood)).any():
                # When the log-likelihood assumes inane values, reinitialize
                # model
                select = torch.randint(
                    self.N, size=(self.T * self.n_components, ))
                mu_init = x.reshape(-1, self.n_features)[select, :].view(
                    self.T, self.n_components, self.n_features)
                self._init_params(mu_init=mu_init)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if torch.is_tensor(j) and (j <= delta).any():
                # When score decreases, revert to old parameters
                t = (j <= delta)
                mu_old = t.float().view(self.T, 1, 1) * mu_old + (
                    ~t).float().view(self.T, 1, 1) * self.mu
                var_old = t.float().view(self.T, 1, 1, 1) * var_old + (
                    ~t).float().view(self.T, 1, 1, 1) * self.var
                self.update_mu(mu_old)
                self.update_var(var_old)

        self.params_fitted = True

    def estimate_log_prob(self, x):
        """Estimate the log-likelihood probability that samples belong to the
        k-th Gaussian.

        Args:
            x (torch.Tensor): (T, n, d) or (T, n, 1, d)

        Returns:
            torch.Tensor: log-likelihood probability that samples belong to \
                the k-th Gaussian with dimensions (T, n, k, 1).
        """
        x = self.check_size(x)

        mu = self.mu
        var = self.var
        inverse_var = torch.inverse(var)
        d = x.shape[-1]
        log_2pi = d * np.log(2. * pi)
        det_var = torch.det(var)
        log_det = torch.log(det_var).view(self.T, 1, self.n_components, 1)
        log_det[log_det == -np.inf] = self.lower_bound_logdet
        mu = mu.unsqueeze(1)
        x_mu_T = (x - mu).unsqueeze(-2)
        x_mu = (x - mu).unsqueeze(-1)
        x_mu_T_inverse_var = x_mu_T.matmul(inverse_var.unsqueeze(1))
        x_mu_T_inverse_var_x_mu = x_mu_T_inverse_var.matmul(x_mu).squeeze(-1)
        log_p = -.5 * (log_2pi + log_det + x_mu_T_inverse_var_x_mu)

        return log_p

    def log_resp_step(self, x):
        """Computes log-responses that indicate the (logarithmic) posterior
        belief (sometimes called responsibilities) that a data point was
        generated by one of the k mixture components. Also returns the mean of
        the mean of the logarithms of the probabilities (as is done in
        sklearn). This is the so-called expectation step of the EM-algorithm.

        Args:
            x (torch.Tensor): (T, n, d) or (T, n, 1, d)

        Returns:
            tuple:

                log_prob_norm (torch.Tensor): the mean of the mean of the \
                    logarithms of the probabilities.
                log_resp (torch.Tensor): log-responses that indicate the \
                    posterior belief.
        """
        x = self.check_size(x)

        weighted_log_prob = self.estimate_log_prob(x) + torch.log(
            self.pi).unsqueeze(1)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=2, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm, dim=(1, 2)), log_resp

    def EM_step(self, x, log_resp):
        """From the log-probabilities, computes new parameters pi, mu, var
        (that maximize the log-likelihood). This is the maximization step of
        the EM-algorithm.

        Args:
            x (torch.Tensor): (T, n, d) or (T, n, 1, d)
            log_resp (torch.Tensor): (T, n, k, 1)

        Returns:
            tuple:

                pi (torch.Tensor): (T, k, 1)
                mu (torch.Tensor): (T, k, d)
                var (torch.Tensor): (T, k, d) or (T, k, d, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=1) + self.eps
        mu = torch.sum(resp * x, dim=1) / pi

        eps = (torch.eye(self.n_features) * self.eps).to(x.device)
        var = torch.sum((x - mu.unsqueeze(1)).unsqueeze(-1).matmul(
            (x - mu.unsqueeze(1)).unsqueeze(-2)) *
                        resp.unsqueeze(-1), dim=1) / \
            torch.sum(resp, dim=1).unsqueeze(-1) + eps

        pi = pi / x.shape[1]

        return pi, mu, var

    def em_runner(self, x):
        """Performs one iteration of the expectation-maximization algorithm by
        calling the respective subroutines.

        Args:
            x (torch.Tensor): (n, 1, d)
        """
        _, log_resp = self.log_resp_step(x)
        pi, mu, var = self.EM_step(x, log_resp)

        self.update_pi(pi)
        self.update_mu(mu)
        self.update_var(var)

    def get_score(self, x, sum_data=True):
        """Computes the log-likelihood of the data under the model.

        Args:
            x (torch.Tensor): (T, n, 1, d)
            sum_data (bool,optional): Flag of whether to sum scores.

        Returns:
            torch.Tensor: score or per_sample_score.
        """
        weighted_log_prob = self.estimate_log_prob(x) + torch.log(
            self.pi).unsqueeze(1)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=2)

        if sum_data:
            return per_sample_score.sum(dim=1)
        else:
            return per_sample_score.squeeze(-1)

    def update_mu(self, mu):
        """Updates mean to the provided value.

        Args:
            mu (torch.Tensor):
        """
        assert mu.size() == (
            self.T, self.n_components, self.n_features
        ), 'Input mu does not have required tensor dimensions (%i, %i, %i)' % (
            self.T, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (self.T, self.n_components, self.n_features):
            self.mu = mu.clone()

    def update_var(self, var):
        """Updates variance to the provided value.

        Args:
            var (torch.Tensor): (T, k, d) or (T, k, d, d)
        """
        assert var.size() == (self.T, self.n_components, self.n_features,
                              self.n_features), \
            'Input var does not have required tensor' \
            ' dimensions (%i, %i, %i, %i)' % \
            (self.T, self.n_components,
             self.n_features,
             self.n_features)

        if var.size() == (self.n_components, self.n_features, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (self.T, self.n_components, self.n_features,
                            self.n_features):
            self.var = var.clone()

    def update_pi(self, pi):
        """Updates pi to the provided value.

        Args:
            pi (torch.Tensor): (T, k, 1)
        """

        assert pi.size() == (
            self.T, self.n_components, 1
        ), 'Input pi does not have required tensor dimensions (%i, %i, %i)' % (
            self.T, self.n_components, 1)

        self.pi = pi.clone()
