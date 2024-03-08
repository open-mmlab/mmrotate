# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import Module

M_EPS = 1e-16


# This code was copied from
# "https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/ot_loss.py"
class OT_Loss(Module):

    def __init__(self, num_of_iter_in_ot=100, reg=10.0, method='sinkhorn'):
        super(OT_Loss, self).__init__()
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.method = method

    def forward(self,
                t_scores,
                s_scores,
                pts,
                cost_type='all',
                clamp_ot=False,
                aux_cost=None):
        r"""
        Calculating Optimal Transport loss between teacher and
        student's distribution.
        Cost map is defined as: cost = dist(p_t, p_s) + dist(score_t, score_s).
        All dist are l2 distance.
        Args:
            t_scores: Tensor with shape (N, )
            s_scores: Tensor with shape (N, )

        Returns:

        """
        assert cost_type in ['all', 'dist', 'score']
        with torch.no_grad():
            t_scores_prob = torch.softmax(t_scores, dim=0)
            s_scores_prob = torch.softmax(s_scores, dim=0)
            score_cost = (t_scores.detach().unsqueeze(1) -
                          s_scores.detach().unsqueeze(0))**2
            score_cost = score_cost / score_cost.max()
            if cost_type in ['all', 'dist']:
                coord_x = pts[:, 0]
                coord_y = pts[:, 1]
                dist_x = (coord_x.reshape(1, -1) - coord_x.reshape(-1, 1))**2
                dist_y = (coord_y.reshape(1, -1) - coord_y.reshape(-1, 1))**2
                dist_cost = (dist_x + dist_y).to(t_scores_prob.device)
                dist_cost = dist_cost / dist_cost.max()
                if cost_type == 'all':
                    cost_map = dist_cost + score_cost
                else:
                    cost_map = dist_cost
            else:
                cost_map = score_cost
            if not isinstance(aux_cost, type(None)):
                cost_map = cost_map + aux_cost
            # cost_map = (dist_cost + score_cost) / 2
            source_prob = s_scores_prob.detach().view(-1)
            target_prob = t_scores_prob.detach().view(-1)
            if t_scores.shape[0] < 2000:  # 2500
                _, log = self.sinkhorn(
                    target_prob,
                    source_prob,
                    cost_map,
                    self.reg,
                    maxIter=self.num_of_iter_in_ot,
                    log=True,
                    method=self.method)
                beta = log[
                    'beta']  # size is the same as source_prob: [#cood * #cood]
            else:
                _, log = self.sinkhorn(
                    target_prob.cpu(),
                    source_prob.cpu(),
                    cost_map.cpu(),
                    self.reg,
                    maxIter=self.num_of_iter_in_ot,
                    log=True,
                    method=self.method)
                beta = log['beta'].to(
                    target_prob.device
                )  # size is the same as source_prob: [#cood * #cood]
        # compute the gradient of Optimal Transport loss
        # to predicted density (unnormed_density).
        # im_grad = beta / source_count -
        # < beta, source_density> / (source_count)^2
        source_density = s_scores.detach().view(-1)
        source_count = source_density.sum()
        im_grad_1 = (source_count) / (source_count * source_count +
                                      1e-8) * beta  # size of [#cood * #cood]
        im_grad_2 = (source_density * beta).sum() / (
            source_count * source_count + 1e-8)  # size of 1
        im_grad = im_grad_1 - im_grad_2
        im_grad = im_grad.detach()
        # Define loss = <im_grad, predicted density>.
        # The gradient of loss w.r.t prediced density is im_grad.
        if clamp_ot:
            return torch.clamp_min(torch.sum(s_scores * im_grad), 0)
        return torch.sum(s_scores * im_grad)

    # The code below was copied from SOOD
    # (https://github.com/HamPerdredes/SOOD/blob/main/
    # ssad/models/losses/utils/bregman_pytorch.py)
    def sinkhorn(self,
                 a,
                 b,
                 C,
                 reg=1e-1,
                 method='sinkhorn',
                 maxIter=1000,
                 tau=1e3,
                 stopThr=1e-9,
                 verbose=False,
                 log=True,
                 warm_start=None,
                 eval_freq=10,
                 print_freq=200,
                 **kwargs):
        r"""Solve the entropic regularization optimal transport The input
        should be PyTorch tensors The function solves the following
        optimization problem:

        .. math::
            \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                 \gamma^T 1= b
                 \gamma\geq 0
        where :
        - C is the (ns,nt) metric cost matrix
        - :math:`\Omega` is the entropic regularization term :
        math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
        - a and b are target and source measures (sum to 1)
        The algorithm used for solving the problem is the
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

        Parameters
        ----------
        a : torch.tensor (na,)
            samples measure in the target domain
        b : torch.tensor (nb,)
            samples in the source domain
        C : torch.tensor (na,nb)
            loss matrix
        reg : float
            Regularization term > 0
        method : str
            method used for the solver either 'sinkhorn', 'greenkhorn',
            'sinkhorn_stabilized' or 'sinkhorn_epsilon_scaling',
            see those function for specific parameters
        maxIter : int, optional
            Max number of iterations
        stopThr : float, optional
            Stop threshold on error ( > 0 )
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            record log if True

        Returns
        -------
        gamma : (na x nb) torch.tensor
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary return only if log==True in parameters

        References
        ----------
        [1] M. Cuturi, Sinkhorn Distances :
        Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013
        See Also
        --------
        """

        if method.lower() == 'sinkhorn':
            return self.sinkhorn_knopp(
                a,
                b,
                C,
                reg,
                maxIter=maxIter,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warm_start=warm_start,
                eval_freq=eval_freq,
                print_freq=print_freq,
                **kwargs)
        elif method.lower() == 'sinkhorn_stabilized':
            return self.sinkhorn_stabilized(
                a,
                b,
                C,
                reg,
                maxIter=maxIter,
                tau=tau,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warm_start=warm_start,
                eval_freq=eval_freq,
                print_freq=print_freq,
                **kwargs)
        elif method.lower() == 'sinkhorn_epsilon_scaling':
            return self.sinkhorn_epsilon_scaling(
                a,
                b,
                C,
                reg,
                maxIter=maxIter,
                maxInnerIter=100,
                tau=tau,
                scaling_base=0.75,
                scaling_coef=None,
                stopThr=stopThr,
                verbose=False,
                log=log,
                warm_start=warm_start,
                eval_freq=eval_freq,
                print_freq=print_freq,
                **kwargs)
        else:
            raise ValueError("Unknown method '%s'." % method)

    def sinkhorn_knopp(self,
                       a,
                       b,
                       C,
                       reg=1e-1,
                       maxIter=1000,
                       stopThr=1e-9,
                       verbose=False,
                       log=False,
                       warm_start=None,
                       eval_freq=10,
                       print_freq=200,
                       **kwargs):
        r"""Solve the entropic regularization optimal transport The input
        should be PyTorch tensors The function solves the following
        optimization problem:

        .. math::
            \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                 \gamma^T 1= b
                 \gamma\geq 0
        where :
        - C is the (ns,nt) metric cost matrix
        - :math:`\Omega` is the entropic regularization term :
        math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
        - a and b are target and source measures (sum to 1)
        The algorithm used for solving the problem is the
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

        Parameters
        ----------
        a : torch.tensor (na,)
            samples measure in the target domain
        b : torch.tensor (nb,)
            samples in the source domain
        C : torch.tensor (na,nb)
            loss matrix
        reg : float
            Regularization term > 0
        maxIter : int, optional
            Max number of iterations
        stopThr : float, optional
            Stop threshold on error ( > 0 )
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            record log if True

        Returns
        -------
        gamma : (na x nb) torch.tensor
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary return only if log==True in parameters

        References
        ----------
        [1] M. Cuturi, Sinkhorn Distances :
        Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013
        See Also
        --------
        """

        device = a.device
        na, nb = C.shape

        assert na >= 1 and nb >= 1, 'C needs to be 2d'
        assert na == a.shape[0] and nb == b.shape[
            0], "Shape of a or b doesn't match that of C"
        assert reg > 0, 'reg should be greater than 0'
        assert a.min() >= 0. and b.min(
        ) >= 0., 'Elements in a or b less than 0'

        if log:
            log = {'err': []}

        if warm_start is not None:
            u = warm_start['u']
            v = warm_start['v']
        else:
            u = torch.ones(na, dtype=a.dtype).to(device) / na
            v = torch.ones(nb, dtype=b.dtype).to(device) / nb

        K = torch.empty(C.shape, dtype=C.dtype).to(device)
        torch.div(C, -reg, out=K)
        torch.exp(K, out=K)

        b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

        it = 1
        err = 1

        # allocate memory beforehand
        KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
        Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

        while (err > stopThr and it <= maxIter):
            upre, vpre = u, v

            KTu = torch.matmul(u, K).squeeze(0)
            v = torch.div(b, KTu + M_EPS)
            Kv = torch.matmul(K, v).squeeze(0)
            u = torch.div(a, Kv + M_EPS)

            # torch.matmul(u, K, out=KTu.unsqueeze(0))
            # v = torch.div(b, KTu + M_EPS)
            # torch.matmul(K, v, out=Kv)
            # u = torch.div(a, Kv + M_EPS)

            if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                    torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
                print('Warning: numerical errors at iteration', it)
                u, v = upre, vpre
                break

            if log and it % eval_freq == 0:
                # we can speed up the process
                # by checking for the error only all
                # the eval_freq iterations
                # below is equivalent to:
                # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
                # but with more memory efficient
                b_hat = torch.matmul(u, K) * v
                err = (b - b_hat).pow(2).sum().item()
                # err = (b - b_hat).abs().sum().item()
                log['err'].append(err)

            if verbose and it % print_freq == 0:
                print('iteration {:5d}, constraint error {:5e}'.format(
                    it, err))

            it += 1

        if log:
            log['u'] = u
            log['v'] = v
            log['alpha'] = reg * torch.log(u + M_EPS)
            log['beta'] = reg * torch.log(v + M_EPS)

        # transport plan
        P = u.reshape(-1, 1) * K * v.reshape(1, -1)
        if log:
            return P, log
        else:
            return P

    def sinkhorn_stabilized(self,
                            a,
                            b,
                            C,
                            reg=1e-1,
                            maxIter=1000,
                            tau=1e3,
                            stopThr=1e-9,
                            verbose=False,
                            log=False,
                            warm_start=None,
                            eval_freq=10,
                            print_freq=200,
                            **kwargs):
        r"""Solve the entropic regularization Optimal Transport problem with
        log stabilization
        The function solves the following optimization problem:

        .. math::
            \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                 \gamma^T 1= b
                 \gamma\geq 0
        where :
        - C is the (ns,nt) metric cost matrix
        - :math:`\Omega` is the entropic regularization term :
        math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
        - a and b are target and source measures (sum to 1)

        The algorithm used for solving the problem is the
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [1]
        but with the log stabilization proposed in [3] an defined in
         [2] (Algo 3.1)

        Parameters
        ----------
        a : torch.tensor (na,)
            samples measure in the target domain
        b : torch.tensor (nb,)
            samples in the source domain
        C : torch.tensor (na,nb)
            loss matrix
        reg : float
            Regularization term > 0
        tau : float
            thershold for max value in u or v for log scaling
        maxIter : int, optional
            Max number of iterations
        stopThr : float, optional
            Stop threshold on error ( > 0 )
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            record log if True

        Returns
        -------
        gamma : (na x nb) torch.tensor
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary return only if log==True in parameters

        References
        ----------
        [1] M. Cuturi, Sinkhorn Distances :
        Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013
        [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems.
        SIAM Journal on Scientific Computing, 2019
        [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

        See Also
        --------
        """

        device = a.device
        na, nb = C.shape

        assert na >= 1 and nb >= 1, 'C needs to be 2d'
        assert na == a.shape[0] and nb == b.shape[
            0], "Shape of a or b doesn't match that of C"
        assert reg > 0, 'reg should be greater than 0'
        assert a.min() >= 0. and b.min(
        ) >= 0., 'Elements in a or b less than 0'

        if log:
            log = {'err': []}

        if warm_start is not None:
            alpha = warm_start['alpha']
            beta = warm_start['beta']
        else:
            alpha = torch.zeros(na, dtype=a.dtype).to(device)
            beta = torch.zeros(nb, dtype=b.dtype).to(device)

        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

        def update_K(alpha, beta):
            """log space computation."""
            """memory efficient"""
            torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=K)
            torch.add(K, -C, out=K)
            torch.div(K, reg, out=K)
            torch.exp(K, out=K)

        def update_P(alpha, beta, u, v, ab_updated=False):
            """log space P (gamma) computation."""
            torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=P)
            torch.add(P, -C, out=P)
            torch.div(P, reg, out=P)
            if not ab_updated:
                torch.add(P, torch.log(u + M_EPS).reshape(-1, 1), out=P)
                torch.add(P, torch.log(v + M_EPS).reshape(1, -1), out=P)
            torch.exp(P, out=P)

        K = torch.empty(C.shape, dtype=C.dtype).to(device)
        update_K(alpha, beta)

        b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

        it = 1
        err = 1
        ab_updated = False

        # allocate memory beforehand
        KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
        Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
        P = torch.empty(C.shape, dtype=C.dtype).to(device)

        while (err > stopThr and it <= maxIter):
            torch.matmul(u, K, out=KTu)
            v = torch.div(b, KTu + M_EPS)
            torch.matmul(K, v, out=Kv)
            u = torch.div(a, Kv + M_EPS)

            ab_updated = False
            # remove numerical problems and store them in K
            if u.abs().sum() > tau or v.abs().sum() > tau:
                alpha += reg * torch.log(u + M_EPS)
                beta += reg * torch.log(v + M_EPS)
                u.fill_(1. / na)
                v.fill_(1. / nb)
                update_K(alpha, beta)
                ab_updated = True

            if log and it % eval_freq == 0:
                # we can speed up the process by checking for
                # the error only all the eval_freq iterations
                update_P(alpha, beta, u, v, ab_updated)
                b_hat = torch.sum(P, 0)
                err = (b - b_hat).pow(2).sum().item()
                log['err'].append(err)

            if verbose and it % print_freq == 0:
                print('iteration {:5d}, constraint error {:5e}'.format(
                    it, err))

            it += 1

        if log:
            log['u'] = u
            log['v'] = v
            log['alpha'] = alpha + reg * torch.log(u + M_EPS)
            log['beta'] = beta + reg * torch.log(v + M_EPS)

        # transport plan
        update_P(alpha, beta, u, v, False)

        if log:
            return P, log
        else:
            return P

    def sinkhorn_epsilon_scaling(self,
                                 a,
                                 b,
                                 C,
                                 reg=1e-1,
                                 maxIter=100,
                                 maxInnerIter=100,
                                 tau=1e3,
                                 scaling_base=0.75,
                                 scaling_coef=None,
                                 stopThr=1e-9,
                                 verbose=False,
                                 log=False,
                                 warm_start=None,
                                 eval_freq=10,
                                 print_freq=200,
                                 **kwargs):
        r"""Solve the entropic regularization Optimal Transport problem
        with log stabilization
        The function solves the following optimization problem:

        .. math::
            \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                 \gamma^T 1= b
                 \gamma\geq 0
        where :
        - C is the (ns,nt) metric cost matrix
        - :math:`\Omega` is the entropic regularization term :
        math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
        - a and b are target and source measures (sum to 1)

        The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
        scaling algorithm as proposed in [1] but with the log stabilization
        proposed in [3] and the log scaling proposed in [2] algorithm 3.2

        Parameters
        ----------
        a : torch.tensor (na,)
            samples measure in the target domain
        b : torch.tensor (nb,)
            samples in the source domain
        C : torch.tensor (na,nb)
            loss matrix
        reg : float
            Regularization term > 0
        tau : float
            thershold for max value in u or v for log scaling
        maxIter : int, optional
            Max number of iterations
        stopThr : float, optional
            Stop threshold on error ( > 0 )
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            record log if True

        Returns
        -------
        gamma : (na x nb) torch.tensor
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary return only if log==True in parameters

        References
        ----------
        [1] M. Cuturi, Sinkhorn Distances :
        Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013
        [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems.
        SIAM Journal on Scientific Computing, 2019
        [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

        See Also
        --------
        """

        na, nb = C.shape

        assert na >= 1 and nb >= 1, 'C needs to be 2d'
        assert na == a.shape[0] and nb == b.shape[
            0], "Shape of a or b doesn't match that of C"
        assert reg > 0, 'reg should be greater than 0'
        assert a.min() >= 0. and b.min(
        ) >= 0., 'Elements in a or b less than 0'

        def get_reg(it, reg, pre_reg):
            if it == 1:
                return scaling_coef
            else:
                if (pre_reg - reg) * scaling_base < M_EPS:
                    return reg
                else:
                    return (pre_reg - reg) * scaling_base + reg

        if scaling_coef is None:
            scaling_coef = C.max() + reg

        it = 1
        err = 1
        running_reg = scaling_coef

        if log:
            log = {'err': []}

        warm_start = None

        while (err > stopThr and it <= maxIter):
            running_reg = get_reg(it, reg, running_reg)
            P, _log = self.sinkhorn_stabilized(
                a,
                b,
                C,
                running_reg,
                maxIter=maxInnerIter,
                tau=tau,
                stopThr=stopThr,
                verbose=False,
                log=True,
                warm_start=warm_start,
                eval_freq=eval_freq,
                print_freq=print_freq,
                **kwargs)

            warm_start = {}
            warm_start['alpha'] = _log['alpha']
            warm_start['beta'] = _log['beta']

            primal_val = (
                C * P).sum() + reg * (P * torch.log(P)).sum() - reg * P.sum()
            dual_val = (_log['alpha'] * a).sum() + (_log['beta'] *
                                                    b).sum() - reg * P.sum()
            err = primal_val - dual_val
            log['err'].append(err)

            if verbose and it % print_freq == 0:
                print('iteration {:5d}, constraint error {:5e}'.format(
                    it, err))

            it += 1

        if log:
            log['alpha'] = _log['alpha']
            log['beta'] = _log['beta']
            return P, log
        else:
            return P
