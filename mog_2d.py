# Implements black-box variational inference, where the variational
# distribution is a mixture of Gaussians.
#
# This trick was written up by Alex Graves in this note:
# http://arxiv.org/abs/1607.05690

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
    return tf.reduce_logsumexp([sigma_density + mu_density, sigma_density2 + mu_density2], axis=0)

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


def log_mean_exp(x, axis=None):
    m = tf.reduce_max(x, axis=axis)
    return m + tf.log(tf.reduce_mean(tf.exp(x - m), axis=axis))

if __name__ == '__main__':
    sess = tf.Session()
    # Specify an inference problem by its unnormalized log-density.
    D = 2
    batch_size = 64
    post = MOGApproximatePosterior(D, batch_size, 10)
    log_ps = unormalized_log_posterior_density(post.sample)
    elbo = tf.reduce_mean(log_ps - post.log_qs)
    opt = tf.train.AdamOptimizer(learning_rate=.1).minimize(-1. * elbo, var_list=post.params)

    init = tf.global_variables_initializer()
    sess.run(init)

    # needed for display purposes
    x = tf.placeholder(tf.float32, [None, D])
    true_post_density = tf.exp(unormalized_log_posterior_density(x))
    approx_post_density = tf.exp(post.log_qs_given_x(x))
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def draw():
        plt.cla()
        plot_isocontours(ax, true_post_density, x, sess)
        plot_isocontours(ax, approx_post_density, x, sess)
        ax.set_autoscale_on(False)
        samples = sess.run(post.sample)
        plt.plot(samples[:, 0], samples[:, 1], 'x')
        plt.draw()
        plt.pause(1.0 / 30.0)

    while True:
        _, lbd = sess.run([opt, elbo])
        print("ELBO: {}".format(lbd))
        draw()
