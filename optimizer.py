# optimizer.py
# contains classes for the ADAM and oLNAQ optimizers, which when passed a batch of training data
# and a model will compute necessary forward and backward passes on the model to compute new weights
# for the model by the optimization algorithm

import tensorflow as tf
import utils

class Adam:
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        # Initialize variables on the first call
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var)
            self.s_dvar[i].assign(self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var))
            v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
            var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
        self.t += 1.
        return

    def train_step(self, x_batch, y_batch, loss, acc, model):
        # Update the model state given a batch of data
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            batch_loss = loss(y_pred, y_batch)
        batch_acc = acc(y_pred, y_batch)
        grads = tape.gradient(batch_loss, model.variables)
        self.apply_gradients(grads, model.variables)
        return batch_loss, batch_acc

class oLNAQ:
    def __init__(self, momentum=0.85, memory_size=4, epsilon=1e-10, pscale=1, initial_step=1):
        # initialize optimizer parameters and variable slots
        self.momentum = momentum
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.pscale = pscale
        self.initial_step = initial_step
        self.prev_ps = []
        self.prev_qs = []
        self.alpha = 0
        self.prev_step = []
        self.current_direction = []
        self.current_step = []
        self.current_weights = []
        self.k = 0
        self.built = False

    def train_step(self, x_batch, y_batch, loss, acc, model):
        # initialize previous pairs and previous direction to 0s
        if not self.built:
            y_pred = model(x_batch)
            for var in model.variables:
                v_prev = tf.Variable(tf.zeros(shape=var.shape))
                g = tf.Variable(tf.zeros(shape=var.shape))
                v_cur = tf.Variable(tf.zeros(shape=var.shape))
                w = tf.Variable(tf.zeros(shape=var.shape))
                self.prev_step.append(v_prev)
                self.current_direction.append(g)
                self.current_step.append(v_cur)
                self.current_weights.append(w)
            for i in range(self.memory_size):
                zeros = []
                for var in model.variables:
                    v = tf.Variable(tf.zeros(shape=var.shape))
                    zeros.append(v)
                self.prev_ps.append(zeros)
                self.prev_qs.append(zeros)
            self.built = True

        # step 3 of alg 1
        for i, var in enumerate(model.variables):
            self.current_weights[i].assign(var)
            var.assign_add(self.momentum*self.prev_step[i])
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            batch_loss = loss(y_pred, y_batch)
        grads_1 = tape.gradient(batch_loss, model.variables)
        for i, var in enumerate(model.variables):
            var.assign(self.current_weights[i])

        # 2 loop recursion
        # step 1 of alg 2
        for i, grad in enumerate(grads_1):
            self.current_direction[i].assign(-1*grad)

        beta = []

        # steps 2-5 of alg 2
        for i in range(self.memory_size):
            if i < self.k:
                numerator = utils.inner_product(self.prev_ps[i], self.current_direction)
                denominator = utils.inner_product(self.prev_ps[i], self.prev_qs[i])
                beta.append(numerator / denominator)
                for j, q in enumerate(self.prev_qs[i]):
                    self.current_direction[j].assign_sub(beta[i]*q)

        # steps 6-10 of alg 2
        if self.k == 0:
            for direction in self.current_direction:
                direction.assign(self.epsilon*direction)
        else:
            scale = 0
            num = 0
            for i in range(self.memory_size):
                if i < self.k:
                    numerator = utils.inner_product(self.prev_ps[i], self.prev_qs[i])
                    denominator = utils.inner_product(self.prev_qs[i], self.prev_qs[i])
                    scale += numerator / denominator
                    num += 1.0
            scale = scale / num
            for direction in self.current_direction:
                direction.assign(scale*direction)

        # steps 11-14 of alg 2
        for i in range(self.memory_size - 1, -1, -1):
            if i < self.k:
                numerator = utils.inner_product(self.prev_qs[i], self.current_direction)
                denominator = utils.inner_product(self.prev_qs[i], self.prev_ps[i])
                tau = numerator / denominator
                for j, q in enumerate(self.prev_ps[i]):
                    self.current_direction[j].assign_add((beta[i]-tau)*q)

        # step 5 of alg 1
        full_flattened = tf.zeros([0])
        for direction in self.current_direction:
            flattened = tf.reshape(direction, [-1])
            full_flattened = tf.concat([full_flattened, flattened], axis=0)
        direction_norm = tf.norm(full_flattened)
        for direction in self.current_direction:
            direction.assign(direction / direction_norm)

        # step 6 of alg 1
        learning_rate = self.initial_step / tf.sqrt(float(self.k + 1))
        # step 7-8 of alg 1
        for i, var in enumerate(model.variables):
            self.current_step[i].assign(self.momentum*self.prev_step[i] + learning_rate*self.current_direction[i])
            var.assign_add(self.current_step[i])

        # step 9 of alg 1
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            batch_loss = loss(y_pred, y_batch)
        grads_2 = tape.gradient(batch_loss, model.variables)
        batch_acc = acc(y_pred, y_batch)

        # step 10-12 of alg 1
        for i in range(self.memory_size - 1, 0, -1):
            self.prev_ps[i] = self.prev_ps[i - 1]
            self.prev_qs[i] = self.prev_qs[i - 1]
        for i, var in enumerate(model.variables):
            self.prev_ps[0][i].assign(var - (self.current_weights[i] + self.momentum*self.prev_step[i]))
            self.prev_qs[0][i].assign(grads_2[i] - grads_1[i] + self.pscale*self.prev_ps[0][i])

        # step 13 of alg 1
        self.prev_step = self.current_step
        self.k += 1

        return batch_loss, batch_acc
