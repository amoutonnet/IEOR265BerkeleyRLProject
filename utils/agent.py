import tensorflow as tf


DELTA = 1e-10


class Agent():
    def __init__(self,
                 state_space_shape,              # The shape of the state space
                 action_space_size,              # The size of the action space
                 gamma=0.99,                     # The discounting factor
                 hidden_conv_layers=[],          # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],       # A list of parameters of for each hidden dense layer
                 initializer='random_normal',
                 verbose=False                   # A live status of the training
                 ):
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.hidden_dense_layers = hidden_dense_layers
        self.hidden_conv_layers = hidden_conv_layers
        self.initializer = initializer
        self.verbose = verbose
        self.main_name = None

    def create_conv_layers(self, x, name):
        name = '%s_%s' % (self.main_name, name)
        if len(self.state_space_shape) > 1:
            # Hidden Conv layers, relu activated
            for id_, c in enumerate(self.hidden_conv_layers):
                x = tf.keras.layers.Conv1D(filters=c[0],
                                           kernel_size=c[1],
                                           strides=c[2],
                                           padding='same',
                                           activation='relu',
                                           kernel_initializer=self.initializer,
                                           name='%s_conv_%d' % (name, id_),
                                           )(x)
            # We flatten before dense layers
            x = tf.keras.layers.Flatten(name='%s_flatten' % name)(x)
        return x

    def create_dense_layers(self, x, name):
        name = '%s_%s' % (self.main_name, name)
        for id_, h in enumerate(self.hidden_dense_layers):
            x = tf.keras.layers.Dense(units=h,
                                      activation='relu',
                                      kernel_initializer=self.initializer,
                                      name='%s_dense_%d' % (name, id_),
                                      )(x)
        return x

    def build_network(self):
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def learn_off_policy(self):
        raise NotImplementedError

    def learn_on_policy(self):
        raise NotImplementedError

    def predict_action(self, state):
        raise NotImplementedError

    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + DELTA)
        return x
