from keras.models import model_from_json
import json
from Q_learn import Catch
from utils import *
import numpy as np

# Testing
def test(model):
    grid_size = 10
    global last_frame_time
    # plt.ion() #打开交互模式（在console当中是默认交互模式，但是在python脚本当中是默认阻塞模式）
    env = Catch(grid_size)
    c = 0
    last_frame_time = 0
    points = 0

    for _ in range(10):
        # loss = 0.
        env.reset()
        game_over = False
        input_t = env.observe()
        display_screen(3, points, input_t)
        plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
        plt.savefig("%03d.png" % c)
        c += 1
        while not game_over:
            input_tml = input_t
            q = model.predict(input_tml)
            action = np.argmax(q[0])
            input_t, reward, game_over = env.act(action)
            points += reward
            display_screen(action, points, input_t)
            plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
            plt.savefig("%03d.png" % c)
            c += 1
    display_screen(4, points, input_t)

if __name__ == "__main__":
    with open("model.json","r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")
    test(model)