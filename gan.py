import tensorflow as tf


class Generator:
    def __init__(self):
        self.reuse = False

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)

        with tf.variable_scope('g', reuse=self.reuse):
            outputs = tf.layers.dense(inputs=inputs, units=128, activation=tf.nn.relu, name='dense1')
            outputs = tf.layers.dense(inputs=outputs, units=128, activation=tf.nn.relu, name='dense2')
            outputs = tf.layers.dense(inputs=outputs, units=2, name='g_out')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs


class BaseDiscriminator:
    def __init__(self):
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        outputs = tf.convert_to_tensor(inputs)

        with tf.name_scope('b_d'), tf.variable_scope('b_d', reuse=self.reuse):
            outputs = tf.layers.dense(inputs=outputs, units=128, activation=tf.nn.relu, name='dense1')
            outputs = tf.layers.dense(inputs=outputs, units=128, activation=tf.nn.relu, name='dense2')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='b_d')
        return outputs


class VanillaDiscriminator:
    def __init__(self):
        self.reuse = False
        self.base_discriminator = BaseDiscriminator()

    def __call__(self, inputs, training=False, name=''):
        outputs = self.base_discriminator(inputs, training, name)

        with tf.name_scope('v_d'), tf.variable_scope('v_d', reuse=self.reuse):
            outputs = tf.layers.dense(inputs=outputs, units=1, name='d_out')
            outputs = tf.nn.sigmoid(outputs)

        self.reuse = True
        self.variables = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='v_d'),
                          self.base_discriminator.variables]
        return outputs


class GAN:
    def __init__(self, batch_size=1000, z_dim=256):
        self.batch_size = batch_size
        self.z_dim = z_dim

        self.g = Generator()
        self.d = VanillaDiscriminator()

        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def forward(self, traindata):
        generated = self.g(self.z, training=True)

        g_outputs = self.d(generated, training=True, name='g')
        t_outputs = self.d(traindata, training=True, name='t')

        # loss function
        # OLD : -log(g)
        # g_loss = tf.reduce_mean(-(tf.log(g_outputs)))

        # Proposed : min KL(d|g)
        g_loss = tf.reduce_mean((tf.log(1 - g_outputs) - tf.log(g_outputs)))

        # g_loss = tf.reduce_mean((tf.log(1 - g_outputs)))
        d_loss = tf.reduce_mean(-(tf.log(t_outputs) + tf.log(1 - g_outputs)))

        return {
            self.g: g_loss,
            self.d: d_loss,
        }

    def train(self, losses, learning_rate=0.0002, beta1=0.5):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)

        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    def sample_data(self, inputs=None):
        if inputs is None:
            inputs = self.z

        data = self.g(inputs, training=True)
        return data
