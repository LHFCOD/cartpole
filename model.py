import tensorflow as tf


class Model:
    def __init__(self):
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_q = tf.placeholder(dtype=tf.float32, shape=[None])
        action = tf.cast(tf.expand_dims(self.input_action, axis=-1), tf.float32)
        input = tf.concat([self.input_state, action], axis=-1)
        layer = tf.layers.dense(inputs=input, units=5, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=5, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=5, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=1, activation=None)
        self.output = tf.reshape(layer, shape=[-1])
        self.loss = tf.reduce_mean(tf.square(self.input_q - self.output))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    def train(self, input_state, input_action, input_q):
        return self.sess.run([self.loss, self.train_op], feed_dict={
            self.input_state: input_state,
            self.input_action: input_action,
            self.input_q: input_q
        })

    def eval_q(self, input_state, input_action):
        return self.sess.run(self.output, feed_dict={
            self.input_state: input_state,
            self.input_action: input_action
        })

    def eval_all_q(self, input_state):
        all_q = []
        a1 = [0] * len(input_state)
        q1 = self.eval_q(input_state, a1)

        a2 = [1] * len(input_state)
        q2 = self.eval_q(input_state, a2)

        optimal_action = []
        max_q = []
        for i in range(0, len(input_state)):
            a = 0
            q = q1[i]
            if q2[i] > q1[i]:
                a = 1
                q = q2[i]
            optimal_action.append(a)
            max_q.append(q)
            all_q.append([q1[i], q2[i]])
        return optimal_action, max_q, all_q
