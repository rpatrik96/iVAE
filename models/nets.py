import warnings
from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


from strnn import MaskedLinear
from strnn.models.strNN import OPT_MAP, check_masks


class ConditionedMaskedLinear(MaskedLinear):
    def __init__(self, in_features, out_features, cond_features, init, activation, bias=False, factor_complexity=1):
        super().__init__(in_features, out_features, init, activation)

        self.cond_features = cond_features
        self.factor_complexity = factor_complexity
        self.cond_net1 = nn.Linear(cond_features, self.factor_complexity * self.in_features, bias=True)
        self.cond_net2 = nn.Linear(cond_features, self.factor_complexity * self.out_features, bias=True)
        self.cond_bias = nn.Linear(cond_features, out_features, bias=True)



    def forward(self, x, u):
        # x, u = xu[:, :self.in_features], xu[:, self.in_features:]
        factor1 = self.cond_net1(u).view(-1, self.factor_complexity,
                                         self.in_features)  # .view(x.shape[0], self.out_features, self.in_features)
        factor2 = self.cond_net2(u).view(-1, self.out_features,
                                         self.factor_complexity)  # .view(x.shape[0], self.out_features, self.in_features)

        # (bs, out, in)
        # batched_mask = F.sigmoid(cond_mask)*self.mask
        # einsum (bs, out, in) x (bs, in) -> (bs, out)
        # result = torch.einsum('boi,bi->bo', batched_mask, x)
        # return result +  self.bias + self.cond_bias(u)

        cond_mask = torch.sigmoid(torch.einsum("bfi,bof->boi", factor1, factor2)) * self.mask * self.weight
        # cond_mask= factor1*self.mask *self.weight
        # einsum (bs, out, in) x (bs, in) -> (bs, out)
        result = torch.einsum('boi,bi->bo', cond_mask, x)
        # result = F.linear(x, self.mask * self.weight, self.bias)
        return result

        # return F.linear(x, self.mask *self.weight*m, self.bias )


from strnn.models.model_utils import NONLINEARITIES
from strnn.models.adaptive_layer_norm import AdaptiveLayerNorm


class ConditionalStrNN(nn.Module):
    """Main neural network class that implements a Structured Neural Network.

    Can also become a MADE or Zuko masked NN by specifying the opt_type flag
    """

    def __init__(
            self,
            nin: int,
            hidden_sizes: tuple[int, ...],
            cond_features: int,
            nout: int,
            opt_type: str = "greedy",
            opt_args: dict = {"var_penalty_weight": 0.0},
            precomputed_masks: np.ndarray | None = None,
            adjacency: np.ndarray | None = None,
            activation: str = "relu",
            init_type: str = 'ian_uniform',
            norm_type: str | None = None,
            layer_norm_inverse: bool | None = None,
            init_gamma: float | None = None,
            # min_gamma: float | None = None,
            max_gamma: float | None = None,
            anneal_rate: float | None = None,
            anneal_method: str | None = None,
            wp: float | None = None,
    ):
        """Initialize a Structured Neural Network (StrNN).

        Args:
            nin: input dimension
            hidden_sizes: list of hidden layer sizes
            nout: output dimension
            opt_type: optimization type: greedy, zuko, MADE
            opt_args: additional optimization algorithm params
            precomputed_masks: previously stored masks, use directly
            adjacency: the adjacency matrix, nout by nin
            activation: activation function to use in this NN
            init_type: initialization scheme for weights
            norm_type: normalization type: layer, batch, adaptive_layer
            gamma: temperature parameter for adaptive layer normalization
            wp: weight parameter for adaptive layer normalization
        """
        super().__init__()

        # Set parameters
        self.nin = nin
        self.hidden_sizes = hidden_sizes
        self.nout = nout

        # Define activation
        try:
            self.activation = NONLINEARITIES[activation]
        except ValueError:
            raise ValueError(f"{activation} is not a valid activation!")

        # Set up initialization and normalization schemes
        self.init_type = init_type
        self.norm_type = norm_type
        self.layer_norm_inverse = layer_norm_inverse
        self.init_gamma = init_gamma
        # self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.anneal_rate = anneal_rate
        self.anneal_method = anneal_method
        self.wp = wp

        # Define StrNN network
        self.net_list = []
        hs = [nin] + list(hidden_sizes) + [nout]  # list of all layer sizes

        # Create MaskedLinear and normalizations for each hidden layer
        for h0, h1 in zip(hs[:-1], hs[1:-1]):
            self.net_list.append(ConditionedMaskedLinear(h0, h1, cond_features, self.init_type, activation))

            # Add normalization layer
            if norm_type == 'layer':
                self.net_list.append(nn.LayerNorm(h1))
            elif norm_type == 'batch':
                self.net_list.append(nn.BatchNorm1d(h1))
            elif norm_type == 'adaptive_layer':
                self.net_list.append(AdaptiveLayerNorm(self.wp, self.layer_norm_inverse))
            else:
                if norm_type is not None:
                    raise ValueError(f"Invalid normalization type: {norm_type}")

            # Add the activation function
            self.net_list.append(NONLINEARITIES[activation])

        # Last layer: no normalization or activation
        self.net_list.append(ConditionedMaskedLinear(hs[-2], hs[-1], cond_features, self.init_type, activation))

        self.net = nn.Sequential(*self.net_list)

        # Load adjacency matrix
        self.opt_type = opt_type.lower()
        self.opt_args = opt_args

        if adjacency is not None:
            self.A = adjacency
        else:
            if self.opt_type == "made":
                # Initialize adjacency structure to fully autoregressive
                warnings.warn(("Adjacency matrix is unspecified, defaulting to"
                               " fully autoregressive structure."))
                self.A = np.tril(np.ones((nout, nin)), -1)
            else:
                raise ValueError(("Adjacency matrix must be specified if"
                                  " factorizer is not MADE."))

        # Setup adjacency factorizer
        try:
            self.factorizer = OPT_MAP[self.opt_type](self.A, self.opt_args)
        except ValueError:
            raise ValueError(f"{opt_type} is not a valid opt_type!")

        self.precomputed_masks = precomputed_masks

        # Update masks
        self.update_masks()

    def forward(self, x: torch.Tensor, u) -> torch.Tensor:
        """Propagates the input forward through the StrNN network.

        Args:
            x: Input of size (sample_size by data_dimensions)
        Returns:
            Output of size (sample_size by output_dimensions)
        """
        # TODO: Add gamma
        for layer in self.net_list:
            if isinstance(layer, ConditionedMaskedLinear):
                x = layer(x, u)

        return x

    def update_masks(self):
        """Update masked linear layer masks to respect adjacency matrix."""
        if self.precomputed_masks is not None:
            # Load precomputed masks if provided
            masks = self.precomputed_masks
        else:
            masks = self.factorizer.factorize(self.hidden_sizes)

        self.masks = masks
        assert check_masks(masks, self.A), "Mask check failed!"

        # For when each input produces multiple outputs
        # e.g. each x_i gives mean and variance for Gaussian density estimation
        if self.nout != self.A.shape[0]:
            # Then nout should be an exact multiple of nin
            assert self.nout % self.nin == 0
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # Set the masks in all MaskedLinear layers
        mask_idx = 0

        mask_so_far = self.masks[0].T
        for layer in self.net:
            if isinstance(layer, ConditionedMaskedLinear):
                layer.set_mask(self.masks[mask_idx])
                if mask_idx > 0:
                    mask_so_far = self.masks[mask_idx].T @ mask_so_far
                mask_idx += 1
            elif isinstance(layer, AdaptiveLayerNorm):
                layer.set_mask(mask_so_far)


def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))


def log_laplace(x, mu, b, broadcast_size=False):
    """compute the log-pdf of a laplace distribution with diagonal covariance"""
    # b might not have batch_dimension. This case is handled by _check_inputs
    if broadcast_size:
        mu, b = _check_inputs(x.size(), mu, b)
    else:
        mu, b = _check_inputs(None, mu, b)
    return -torch.log(2 * b) - (x - mu).abs().div(b)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class ResidualStrNN(nn.Module):
    def __init__(self, strnn, mlp):
        super().__init__()
        self.strnn = strnn
        self.mlp = mlp

    def forward(self, x, u):
        mlp_u = self.mlp(u)
        return self.strnn(x) + mlp_u


class cleanIVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1,
                 use_strnn=False, separate_aux=False, residual_aux=False, use_chain=False,
                 strnn_layers=1, strnn_width=40, aux_net_layers=1, ignore_u=False, cond_strnn=False, adjacency=None, obs_layers=None):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior params
        self.prior_mean = torch.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params

        self.separate_aux = separate_aux
        self.residual_aux = residual_aux
        self.use_strnn = use_strnn
        self.use_chain = use_chain
        self.strnn_layers = strnn_layers
        self.strnn_width = strnn_width
        self.ignore_u = ignore_u
        self.cond_strnn = cond_strnn
        self.obs_layers = obs_layers
        self.aux_net_layers = aux_net_layers

        self._setup_obs_unmixing()
        self._setup_encoder(adjacency)

    def _setup_obs_unmixing(self):
        if self.obs_layers is not None:
            obs_unmixing = []

            if self.obs_layers >= 2:
                obs_unmixing.append(
                    nn.Linear(self.data_dim, self.hidden_dim, bias=False)
                )
                obs_unmixing.append(nn.LeakyReLU(negative_slope=0.25))
                for _ in range(0, self.obs_layers - 2):
                    obs_unmixing.append(
                        nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                    )
                    obs_unmixing.append(nn.LeakyReLU(negative_slope=0.25))

                obs_unmixing.append(
                    nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
                )
                obs_unmixing.append(nn.LeakyReLU(negative_slope=0.25))
            else:
                obs_unmixing.append(
                    nn.Linear(self.data_dim, self.latent_dim, bias=False)
                )
                obs_unmixing.append(nn.LeakyReLU(negative_slope=0.25))

            self.obs_unmixing = nn.Sequential(
                *obs_unmixing,
            )
        else:
            self.obs_unmixing = None

    def _setup_encoder(self, adjacency):
        in_dim = self.data_dim if self.obs_unmixing is None else self.latent_dim
        if self.use_strnn is False:
            if self.ignore_u is False:
                self.z_mean = MLP(in_dim + self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, activation=self.activation,
                                  slope=self.slope)
            else:
                self.z_mean = MLP(in_dim, self.latent_dim, self.hidden_dim, self.n_layers, activation=self.activation, slope=self.slope)
        else:
            print("----------------------")
            print(f"Using {'conditional' if self.cond_strnn is True else ''} StrNN")
            if adjacency is not None:
                print(f"Using {adjacency=}")
            print("----------------------")
            print(f"{self.aux_dim=}")

            hidden_sizes = [
                self.strnn_width for _ in range(self.strnn_layers)
            ]

            from strnn.models.strNN import StrNN

            if self.cond_strnn is True:

                if adjacency is None:
                    adjacency = torch.tril(
                        torch.ones(self.latent_dim,
                                   self.latent_dim)
                    ).numpy()

                    # make it a chain
                    if self.use_chain:
                        adjacency = np.tril(adjacency.T, k=1).T

                strnn = ConditionalStrNN(
                    nin=in_dim,
                    hidden_sizes=(tuple(hidden_sizes)),
                    cond_features=self.aux_dim,
                    nout=self.latent_dim,
                    opt_type="greedy",
                    adjacency=adjacency,
                    activation="leaky_relu",
                    init_type="ian_uniform",
                    norm_type="batch",
                )

                self.z_mean = strnn

            else:
                if self.separate_aux is False:
                    if adjacency is None:
                        adjacency = torch.tril(
                            torch.ones(in_dim + self.aux_dim,
                                       self.latent_dim + self.aux_dim)
                        ).numpy()

                        adjacency[:, self.latent_dim:] = 1.
                        adjacency[self.latent_dim:, :] = 1.

                    hidden_sizes = [
                        (1 * self.hidden_dim) for _ in range(3)
                    ]

                    strnn = StrNN(
                        nin=in_dim + self.aux_dim,
                        hidden_sizes=(tuple(hidden_sizes)),
                        nout=self.latent_dim + self.aux_dim,
                        opt_type="greedy",
                        adjacency=adjacency,
                        activation="leaky_relu",
                        init_type="ian_uniform",
                        # norm_type="batch",
                    )
                    self.z_mean = strnn
                else:

                    if adjacency is None:
                        adjacency = torch.tril(
                            torch.ones(in_dim,
                                       self.latent_dim)
                        ).numpy()

                        # make it a chain
                        if self.use_chain:
                            adjacency = np.tril(adjacency.T, k=1).T

                    if self.ignore_u is False:

                        if self.residual_aux:
                            aux_net = MLP(self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, activation=self.activation, slope=self.slope)

                        else:
                            aux_net = MLP(self.aux_dim +in_dim, self.latent_dim, self.hidden_dim, self.aux_net_layers,
                                          activation=self.activation, slope=self.slope)

                        strnn = StrNN(
                            nin=in_dim,
                            hidden_sizes=(tuple(hidden_sizes)),
                            nout=self.latent_dim,
                            opt_type="greedy",
                            adjacency=adjacency,
                            activation="leaky_relu",
                            init_type="ian_uniform",
                            norm_type="batch",
                        )

                        if self.residual_aux:
                            self.z_mean = ResidualStrNN(strnn, aux_net)
                        else:
                            self.z_mean = nn.Sequential(*[aux_net, strnn])
                    else:
                        self.z_mean = StrNN(
                            nin=in_dim,
                            hidden_sizes=(tuple(hidden_sizes)),
                            nout=self.latent_dim,
                            opt_type="greedy",
                            adjacency=adjacency,
                            activation="leaky_relu",
                            init_type="ian_uniform",
                            norm_type="batch",
                        )
        if self.ignore_u is False:
            self.z_log_var = MLP(in_dim + self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, activation=self.activation,
                                 slope=self.slope)
        else:
            self.z_log_var = MLP(in_dim, self.latent_dim, self.hidden_dim, self.n_layers, activation=self.activation, slope=self.slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):

        xu = torch.cat((x, u), 1).float()
        if self.use_strnn is False:
            if self.ignore_u is False:
                z_mean = self.z_mean(xu)
            else:
                z_mean = self.z_mean(x.float())
        else:
            if self.cond_strnn is True:
                z_mean = self.z_mean(x.float(), u)
            else:
                if self.ignore_u is False:
                    if self.separate_aux is False:
                        z_mean = self.z_mean(xu)[:, :self.latent_dim]
                    else:
                        if self.residual_aux:
                            z_mean = self.z_mean(x, u)
                        else:
                            z_mean = self.z_mean(xu)
                else:
                    z_mean = self.z_mean(x.float())

        if self.ignore_u is False:
            logv = self.z_log_var(xu)
        else:
            logv = self.z_log_var(x.float())
        return z_mean, logv.exp()

    def decoder(self, z_hat):
        x_hat = self.f(z_hat)
        return x_hat

    def prior(self, u):
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        if self.obs_unmixing is not None:
            z_hat = self.obs_unmixing(x)
            x = z_hat
        else:
            z_hat = None
        s_mean, s_var = self.encoder(x, u)
        s_hat = self.reparameterize(s_mean, s_var)
        x_hat = self.decoder(s_hat)
        return x_hat, s_mean, s_var, s_hat, l, z_hat

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        x_hat, s_mean, s_var, s_hat, l, z_hat = self.forward(x, u)
        M, d_latent = s_hat.size()
        logpx = log_normal(x, x_hat, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(s_hat, s_mean, s_var).sum(dim=-1)
        logps_cu = log_normal(s_hat, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(s_hat.view(M, 1, d_latent), s_mean.view(1, M, d_latent), s_var.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
        return elbo, s_hat, z_hat


class cleanVAE(nn.Module):

    def __init__(self, data_dim, latent_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1)
        self.prior_var = torch.ones(1)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def forward(self, x):
        g, v = self.encoder(x)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z = self.forward(x)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps = log_normal(z, None, None, broadcast_size=True).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps)).mean()
        return elbo, z


class Discriminator(nn.Module):
    def __init__(self, z_dim=5, hdim=1000):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, 2),
        )
        self.hdim = hdim

    def forward(self, z):
        return self.net(z).squeeze()


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class Bernoulli(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.bernoulli.Bernoulli(0.5 * torch.ones(1).to(self.device))
        self.name = 'bernoulli'

    def sample(self, p):
        eps = self._dist.sample(p.size())
        return eps

    def log_pdf(self, x, f, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            f = f.view(param_shape)
        lpdf = x * torch.log(f) + (1 - x) * torch.log(1 - f)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


class DiscreteIVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim,
                 n_layers=2, hidden_dim=20, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv

    def decoder_params(self, z):
        f = self.f(z)
        return torch.sigmoid(f)

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.reparameterize(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def elbo(self, x, u):
        f, (g, logv), z, (h, logl) = self.forward(x, u)
        BCE = -F.binary_cross_entropy(f, x, reduction='sum')
        l = logl.exp()
        v = logv.exp()
        KLD = -0.5 * torch.sum(logl - logv - 1 + (g - h).pow(2) / l + v / l)
        return (BCE + KLD) / x.size(0), z  # average per batch


class VAE(nn.Module):

    def __init__(self, latent_dim, data_dim, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder
        self.prior_dist = Normal(device)
        self.prior_mean = torch.zeros(1).to(device)
        self.prior_var = torch.ones(1).to(device)

        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)

    def encoder_params(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def forward(self, x):
        encoder_params = self.encoder_params(x)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, (self.prior_mean, self.prior_var)

    def elbo(self, x):
        decoder_params, encoder_params, z, prior_params = self.forward(x)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_x = self.encoder_dist.log_pdf(z, *encoder_params)
        log_pz = self.prior_dist.log_pdf(z, *prior_params)

        return (log_px_z + log_pz - log_qz_x).mean(), z


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, data_dim,
                 n_layers=2, hidden_dim=20, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

    def encoder_params(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv

    def decoder_params(self, z):
        f = self.f(z)
        return torch.sigmoid(f)

    def forward(self, x):
        encoder_params = self.encoder_params(x)
        z = self.reparameterize(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def elbo(self, x):
        f, (g, logv), z = self.forward(x)
        BCE = -F.binary_cross_entropy(f, x, reduction='sum')
        v = logv.exp()
        KLD = 0.5 * torch.sum(logv + 1 - g.pow(2) - v)
        return (BCE + KLD) / x.size(0), z  # average per batch


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Normal(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()


class LaplaceMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Laplace(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()


class ModularIVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior = GaussianMLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                     device=device, fixed_mean=0)
        else:
            self.prior = prior

        if decoder is None:
            self.decoder = GaussianMLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                       device=device, fixed_var=.01)
        else:
            self.decoder = decoder

        if encoder is None:
            self.encoder = GaussianMLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation,
                                       slope=slope, device=device)
        else:
            self.encoder = encoder

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def forward(self, x, u):
        encoder_params = self.encoder(x, u)
        z = self.encoder.sample(*encoder_params)
        decoder_params = self.decoder(z)
        prior_params = self.prior(u)
        return decoder_params, encoder_params, prior_params, z

    def elbo(self, x, u):
        decoder_params, encoder_params, prior_params, z = self.forward(x, u)
        log_px_z = self.decoder.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder.log_pdf(z, *encoder_params)
        log_pz_u = self.prior.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder.log_pdf(z.view(M, 1, self.latent_dim), encoder_params, reduce=False,
                                              param_shape=(1, M, self.latent_dim))
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z
        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder.log_var(0).exp().item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


# MNIST MODELS - OLD IMPLEMENTATION

class ConvolutionalVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc1 = nn.Linear(200, latent_dim)
        self.fc2 = nn.Linear(200, latent_dim)

        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 1x28x28
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28)).squeeze()
        return self.fc1(F.relu(h)), self.fc2(F.relu(h))

    def decode(self, z):
        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 28 * 28))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        f = self.decode(z)
        return f, mu, logv


class VAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvolutionalIVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc1 = nn.Linear(200, latent_dim)
        self.fc2 = nn.Linear(200, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 1x28x28
        )

        # prior
        self.l1 = nn.Linear(10, 200)
        self.l21 = nn.Linear(200, latent_dim)
        self.l22 = nn.Linear(200, latent_dim)

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28)).squeeze()
        return self.fc1(F.relu(h)), self.fc2(F.relu(h))

    def decode(self, z):
        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 28 * 28))

    def prior(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logv = self.encode(x)
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logv)
        f = self.decode(z)
        return f, mu, logv, mup, logl


class iVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

        hidden_dim = 200
        self.l1 = nn.Linear(10, hidden_dim)
        self.l21 = nn.Linear(hidden_dim, latent_dim)
        self.l22 = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def prior(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784))
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, mup, logl
