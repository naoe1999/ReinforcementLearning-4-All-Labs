import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


class DQN:
    def __init__(self, input_shape, output_size, l_rate=1e-4, name='main'):
        self.input_shape = input_shape
        self.output_size = output_size
        self.net_name = name
        self._build_network(l_rate=l_rate)

    def _build_network(self, l_rate=1e-1):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, padding='same', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.output_size))
        model.compile(optimizer=Adam(learning_rate=l_rate), loss='mse')
        self.net = model

    def copy_to(self, target_dqn):
        target_dqn.net.set_weights(self.net.get_weights())

    def predict(self, state):
        state = state / 255.
        x = np.expand_dims(state, axis=0)
        return self.net.predict(x)

    def update(self, x, y):
        x = x / 255.
        hist = self.net.fit(x, y, epochs=1, shuffle=False)
        loss = hist.history['loss'][0]
        return loss

