"""
Variational Mixture of Gaussians Example

the posterior was lovingly ripped
from https://github.com/HIPS/autograd/blob/master/examples/mixture_variational_inference.py
as well as some of the visualization <3
"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from approx_posteriors import *


def unormalized_log_posterior_density(x):
    mu, log_sigma = x[:, 0:1], x[:, 1:]
    sigma_density = gaussian_log_density(log_sigma, 0, 2. * tf.log(1.35))
    mu_density = gaussian_log_density(mu, -.5, 2. * log_sigma)
    sigma_density2 = gaussian_log_density(log_sigma, .1, 2. * tf.log(1.35))
    mu_density2 = gaussian_log_density(mu, .5, 2. * log_sigma)
    return tf.reduce_logsumexp([sigma_density + mu_density, sigma_density2 + mu_density2], axis=0)[:, 0]


# def unormalized_log_posterior_density(x):
#     """An example 2D intractable distribution:
#     a Gaussian evaluated at zero with a Gaussian prior on the log-variance.
#     x should be N x 2"""
#     mu, log_sigma = x[:, 0:1], x[:, 1:]
#     prior = gaussian_log_density(log_sigma, 0, 2. * tf.log(1.35))
#     likelihood  = gaussian_log_density(mu, 0, 2. * log_sigma)
#     return (prior + likelihood)[:, 0]

# Set up plotting code
def plot_isocontours(ax, func, pl, sess, xlimits=[-4, 4], ylimits=[-4, 4],
                     numticks=101, cmap=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    input = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    zs = sess.run(func, feed_dict={pl: input})
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])


def reduce_logmeanexp(input_tensor, axis=0, keep_dims=False):
    """Computes log(mean(exp(elements across dimensions of a tensor))).
    Args:
      input_tensor: tf.Tensor.
          The tensor to reduce. Should have numeric type.
      axis: int or list of int, optional.
          The dimensions to reduce. If `None` (the default), reduces all
          dimensions.
      keep_dims: bool, optional.
          If true, retains reduced dimensions with length 1.

    Returns:
        tf.Tensor.
        The reduced tensor.
    """
    logsumexp = tf.reduce_logsumexp(input_tensor, axis, keep_dims)
    input_tensor = tf.convert_to_tensor(input_tensor)
    n = float(input_tensor.shape.as_list()[axis])

    return -tf.log(n) + logsumexp


if __name__ == '__main__':
    sess = tf.Session()
    # Specify an inference problem by its unnormalized log-density.
    D = 2
    k = 3
    batch_size = 100
    post = QAFPosterior(D, batch_size, k)
    log_ps = tf.concat([tf.expand_dims(unormalized_log_posterior_density(post.sample[i, :, :]), 0) for i in range(k)],
                       0)
    sample_elbo = log_ps - post.log_qs
    print(sample_elbo.get_shape().as_list())
    elbo = tf.reduce_mean(reduce_logmeanexp(sample_elbo, axis=0))
    opt = tf.train.AdamOptimizer(learning_rate=.1).minimize(-1. * elbo, var_list=post.params)

    init = tf.global_variables_initializer()
    sess.run(init)

    # needed for display purposes
    x = tf.placeholder(tf.float32, [None, D])
    true_post_density = tf.exp(unormalized_log_posterior_density(x))
    approx_post_densites = []
    for mu, logvar in zip(post.mus, post.logvars):
        pr = tf.exp(diag_gaussian_log_density(x, mu[:1, :], logvar[:1, :]))
        approx_post_densites.append(pr)

    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    plt_samples = tf.reshape(post.sample, [-1, D])


    def draw():
        plt.cla()
        plot_isocontours(ax, true_post_density, x, sess)
        for apd in approx_post_densites:
            plot_isocontours(ax, apd, x, sess)
        ax.set_autoscale_on(False)
        samples = sess.run(plt_samples)
        plt.plot(samples[:, 0], samples[:, 1], 'x')
        plt.draw()
        plt.pause(1.0 / 30.0)


    while True:
        _, lbd = sess.run([opt, elbo])
        print("ELBO: {}".format(lbd))
        draw()
