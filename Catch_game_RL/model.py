"""
Q-learning: 三层的dense network。
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

def baseline_model(grid_size, num_actions, hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation="relu"))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.1), "mse") # 这里的loss就是均方误差
    return model