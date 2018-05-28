import tensorflow as tf
from functools import partial

class LeNetClassifier(Classifier):
    epsilon = 10**-10
    def __init__(self, 
        trainingSet, testSet, 
        learning_rate=.01, 
        training_epochs=40,
        batch_size=128):

        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, 
                                shape=(None, 784), 
                                name='InputData')
        self.y = tf.placeholder(tf.float32, 
                                shape=(None, 10),
                                name='LabelData')
        
        with tf.name_scope('LeNetModel'):
            self.lenet_pred = self.LeNet5_Model(x)
        with tf.name_scope('Loss'):
            self.cost = tf.reduce_mean(-tf.reduce_sum(
                self.y*tf.log(tf.clip_by_value(self.lenet_pred, self.epsilon, 1.0)), 
                reduction_indices=1
            ))
        with tf.name_scope('SGD'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

    @staticmethod
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def bias_variable(shape):
        return tf.Variable(tf.constant(0., shape=shape))

    @staticmethod
    def conv2d(input_layer, W, b, padding, activation, pooling, name):
        conv = tf.nn.conv2d(
            input_layer, W, strides=[1, 1, 1, 1], padding=padding, name=name
        )
        return pooling(activation(conv + b))

    @staticmethod
    def fully_connected(input_layer, n_input, n_output, activation, name):
        W = self.weight_variable((n_input, n_output))
        b = self.bias_variable((n_output,))
        return activation(input_layer @ W + b)

    def LeNet5_Model(self, input_x):
        #reshape the input
        self.layer0 = tf.reshape(input_x, (-1, 28, 28, 1))

        self.layer1 = self.conv2d(
            input_layer = self.layer0,
            W = self.weight_variable((5, 5, 1, 6)),
            b = self.bias_variable((6,)),
            padding = "SAME",
            activation = tf.nn.sigmoid,
            pooling = partial(tf.nn.max_pool, 
                            ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], 
                            padding='VALID'),
            name="Layer1"
        )
        
        self.layer2 = self.conv2d(
            input_layer = self.layer1,
            W = self.weight_variable((5, 5, 6, 16)),
            b = self.bias_variable((16,)),
            padding = "VALID",
            activation = tf.nn.sigmoid,
            pooling = partial(tf.nn.max_pool,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID'),
            name="Layer2"
        )
        
        self.layer3 = self.fully_connected(
            input_layer = tf.contrib.layers.flatten(self.layer2),
            n_input=400,
            n_output=120, 
            activation=tf.nn.relu, 
            name="Layer3"
        )
        
        self.layer4 = self.fully_connected(
            input_layer = self.layer3, 
            n_input=120,
            n_output=84, 
            activation=tf.nn.relu, 
            name="Layer4"
        )
        
        self.output = self.fully_connected(
            input_layer = self.layer4, 
            n_input=84,
            n_output=10, 
            activation=tf.nn.softmax, 
            name="Layer5"
        )

    def fit(self, trainingSet):
        self.trainingSet = trainingSet.copy()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.sess.run(
            [self.optimizer, self.cost],
            feed_dict={self.x: self.trainingSet.X, self.y: self.trainingSet.dummy_labels}
        )
        # TODO: compute the Naives bayes using self.layer2 as input

    def predict(self, testSet):
        self.testSet = testSet.copy()
        self.sess.run(
            self.lenet_pred,
            feed_dict={self.x: self.testSet.X}
        )
        # TODO: compute the Naives bayes using self.layer2 as input



    




