from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp

from autograd import grad
from autograd.optimizers import adam
from autograd.core import primitive

def unpack_gaussian_params(params):
    # Variational dist is a diagonal Gaussian.
    D = np.shape(params)[0] / 2
    mean, log_std = params[:D], params[D:]
    return mean, log_std

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def diag_gaussian_density_from_params(params, x):
    # Variational dist is a diagonal Gaussian.
    D = np.shape(params)[0] / 2
    mean, log_std = params[:D], params[D:]
    return diag_gaussian_log_density(x, mean, log_std)

def sample_diag_gaussian(params, num_samples, rs):
    mean, log_std = unpack_gaussian_params(params)
    D = np.shape(mean)[0]
    return rs.randn(num_samples, D) * np.exp(log_std) + mean

def logmeanexp(x, axis=None, keepdims=False):
    n = x.size if axis is None else x.shape[axis]
    return logsumexp(x, axis, 1.0/n, keepdims)

def sample_q(params, num_samples, rs):
    mean, log_std = unpack_gaussian_params(params)
    D = np.shape(mean)[0]
    samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
    log_qs = diag_gaussian_log_density(samples, mean, log_std)
    return log_qs, samples

def iwae_lower_bound(logprob, params, t, k, num_samples, rs=npr.RandomState(0)):
    """Implements importance weighted autoencoders, by
    Burda, Grosse and Salakhutdinov. arxiv.org/abs/1509.00519"""
    log_qs, samples = sample_q(params, num_samples * k, rs)
    log_ps = logprob(samples, t)
    log_ps = np.reshape(log_ps, (num_samples, -1))
    log_qs = np.reshape(log_qs, (num_samples, -1))
    return np.mean(logmeanexp(log_ps - log_qs, axis=1))

def iwae_sample(logprob, params, t, k, num_samples, rs):
    log_qs, samples = sample_q(params, num_samples * k, rs)
    log_ps = logprob(samples, t)
    log_weights = log_ps - log_qs
    samples = np.reshape(samples, (num_samples, k, D))
    weights = np.exp(np.reshape(log_weights, (num_samples, k)))
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    ixs = np.array([np.random.choice(k, p=weights_row) for weights_row in weights])
    return np.array([samples[i, ix, :] for i, ix in enumerate(ixs)])


def init_gaussian_var_params(D, mean_mean=-1, log_std_mean=-5, scale=0.1, rs=npr.RandomState(0)):
    init_mean    = mean_mean * np.ones(D) + rs.randn(D) * scale
    init_log_std = log_std_mean * np.ones(D) + rs.randn(D) * scale
    return np.concatenate([init_mean, init_log_std])


if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-posterior.
    D = 2
    def log_posterior(x, t):
        """An example 2D intractable distribution:
        a Gaussian evaluated at zero with a Gaussian prior on the log-variance."""
        mu, log_sigma = x[:, 0], x[:, 1]
        prior       = norm.logpdf(log_sigma, 0, 1.35)
        likelihood  = norm.logpdf(mu,        0, np.exp(log_sigma))
        return prior + likelihood

    k=3
    num_samples = 200

    vrs = npr.RandomState(0)
    def objective(params, t):
        return -iwae_lower_bound(log_posterior, params, t, k=k, num_samples=num_samples, rs=vrs)

    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim([-4, 1])
        ax.set_xlim([-2, 2])

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    num_plotting_samples = 51

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

        plt.cla()
        target_distribution = lambda x : np.exp(log_posterior(x, t))
        var_distribution = lambda x: np.exp(diag_gaussian_density_from_params(params, x))
        plot_isocontours(ax, target_distribution)
        plot_isocontours(ax, var_distribution)

        rs = npr.RandomState(0)
        samples = iwae_sample(log_posterior, params, t, k, num_plotting_samples, rs)
        plt.plot(samples[:, 0], samples[:, 1], 'x')

        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    adam(grad(objective), init_gaussian_var_params(D), step_size=0.1, num_iters=2000, callback=callback)
