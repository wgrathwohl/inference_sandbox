"""
Houses Approximate Posterior Classes
"""
import tensorflow as tf
import numpy as np

def diag_gaussian_log_density(x, mu, logvar):
    ms, lvs = mu.get_shape().as_list(), logvar.get_shape().as_list()
    assert len(ms) == len(lvs) == 2, (ms, lvs)
    #assert ms[0] == lvs[0] == 1, (ms, lvs)
    lps = -1. * tf.square(x - mu) / (2. * tf.exp(logvar)) - .5 * (np.log(2. * np.pi) + logvar)
    return tf.reduce_sum(lps, axis=1)


def gaussian_log_density(x, mu, logvar):
    lps = -1. * tf.square(x - mu) / (2. * tf.exp(logvar)) - .5 * (np.log(2. * np.pi) + logvar)
    return lps


class ApproximatePosterior:
    def __init__(self):
        pass

    @property
    def params(self):
        return self._params

    @property
    def sample(self):
        return self._sample

    @property
    def log_qs(self):
        try:
            return self._log_qs
        except:
            self._log_qs = self.log_qs_given_x(self.sample)
            return self._log_qs

    def log_qs_given_x(self, x):
        raise NotImplementedError


class GaussianApproximatePosterior(ApproximatePosterior):
    def __init__(self, D, batch_size):
        self._mu = tf.get_variable(
            "mu", shape=[1, D], dtype=tf.float32,
            initializer=tf.constant_initializer(0.), trainable=True
        )
        self._logvar = tf.get_variable(
            "logvar", shape=[1, D], dtype=tf.float32,
            initializer=tf.constant_initializer(-5.), trainable=True
        )
        eps = tf.random_normal((batch_size, D))
        self._sample = self._mu + eps * tf.exp(self._logvar / 2.0)
        self._params = [self._mu, self._logvar]

    def log_qs_given_x(self, x):
        return diag_gaussian_log_density(x, self._mu, self._logvar)

class QFPosterior(ApproximatePosterior):
    def __init__(self, D, batch_size, k):
        self.proposals = []
        batch_size = batch_size
        self.k = k
        for i in range(k):
            with tf.variable_scope("proposal_{}".format(i)):
                self.proposals.append(GaussianApproximatePosterior(D, batch_size))

        samples = tf.concat([tf.expand_dims(d.sample, 0) for d in self.proposals], 0)
        self._sample = samples
        self._params = []
        for p in self.proposals:
            self._params.extend(p.params)

    def log_qs_given_x(self, x):
        qss = []
        for i, proposal in enumerate(self.proposals):
            qs_cur = tf.expand_dims(proposal.log_qs_given_x(x[i, :, :]), 0)
            qss.append(qs_cur)
        qss = tf.concat(qss, 0)
        return qss

gs = lambda x: x.get_shape().as_list()

class QAFPosterior(ApproximatePosterior):
    def __init__(self, D, batch_size, k, rnn_size=8):
        batch_size = batch_size
        self.rnn = tf.contrib.rnn.BasicRNNCell(rnn_size)

        with tf.variable_scope("proposal_0"):
            p0 = GaussianApproximatePosterior(D, batch_size)

        self.initial_state = tf.zeros([batch_size, self.rnn.state_size])
        state = self.initial_state
        s_cur = p0.sample
        samples = [s_cur]
        self.mus = [p0._mu]
        self.logvars = [p0._logvar]
        for i in range(1, k):
            output, state = self.rnn(s_cur, state)
            mu_i = tf.layers.dense(output, D, name="mu_out", reuse=i>1)
            logvar_i = tf.layers.dense(output, D, name="logvar_out", reuse=i>1)
            self.mus.append(mu_i)
            self.logvars.append(logvar_i)
            eps = tf.random_normal((batch_size, D))
            sample = mu_i + eps * tf.exp(logvar_i / 2.0)
            samples.append(sample)

        samples = tf.concat([tf.expand_dims(s, 0) for s in samples], 0)

        self._sample = samples
        self._params = tf.trainable_variables()

    def log_qs_given_x(self, x):
        qss = []
        for i, (mu, logvar) in enumerate(zip(self.mus, self.logvars)):
            x_i = x[i, :, :]
            qs_cur = diag_gaussian_log_density(x_i, mu, logvar)
            qss.append(tf.expand_dims(qs_cur, 0))
        qss = tf.concat(qss, 0)
        return qss


import random
class MOGApproximatePosterior(ApproximatePosterior):
    def __init__(self, D, batch_size, num_gaussians, equal_weight=False):
        self._mus = [
            tf.get_variable(
                "mu_{}".format(i), shape=[1, D], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(random.random(), .5),
                trainable=True
            ) for i in range(num_gaussians)
        ]
        self._logvars = [
            tf.get_variable(
                "logvar_{}".format(i), shape=[1, D], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(-5, .1),
                trainable=True
            ) for i in range(num_gaussians)
        ]
        self._logweights = tf.get_variable(
            "log_weights", shape=[1, num_gaussians],
            dtype=tf.float32, initializer=tf.constant_initializer(1.0),
            trainable=(not equal_weight)
        )

        log_normalize = lambda x: x - tf.reduce_logsumexp(x)
        eps = tf.random_normal((batch_size, D))
        self._normalized_log_weights = log_normalize(self._logweights)
        inds = tf.one_hot(
            tf.multinomial(self._normalized_log_weights, batch_size)[0, :],
            num_gaussians
        )
        inds = tf.reshape(inds, [batch_size, 1, num_gaussians])
        raw_samples = tf.stack(
            [m + eps * tf.exp(lv / 2.0) for m, lv in zip(self._mus, self._logvars)],
            axis=2
        )

        self._sample = tf.reduce_sum(raw_samples * inds, axis=2)
        self._params = self._mus + self._logvars + [self._logweights]

    def log_qs_given_x(self, x):
        per_gaussian_log_qs = tf.stack(
            [diag_gaussian_log_density(x, m, lv) for m, lv in zip(self._mus, self._logvars)],
            axis=1
        )
        return tf.reduce_logsumexp(self._normalized_log_weights + per_gaussian_log_qs, axis=1)
if __name__ == "__main__":
    #qf = QFPosterior(2, 10, 5)
    qaf = QAFPosterior(2, 10, 5)