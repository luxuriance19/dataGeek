from model import baseline_model
# 用来loading和saving models。
import json
from Q_learn import Catch
from Q_learn import ExperienceReplay
from utils import *
import numpy as np

"""
Q-learning的全部的过程，（这里实际上是训练一个策略policy，也就是policy gradient)
DQN（输入是state，输出是每个action的value值，这里输出的这个value值是对应的在这个状态采取这个动作之后总共获得的value的值）
DQN-输入：state，输出：在这个state能够采取不同的action能够获得的最大的value值

这里类似与利用model作为一个simulator（输入是state，输出是action，也就是对DQN的输出取value最大的action），
在simulator的过程当中为了防止只依赖过去的行为而导致之后的行为和之前的行为高度相关，除了利用这个model来做动作的预测之外，
还添加了一些西奥的抖动，也就是随机选取一些动作的过程。这些模拟的动作的过程都存储到experience replay当中（memory当中）

在训练的时候，每次从experience_replay当中取值，目标值就是最大的从当前这个状态开始到游戏结束所获得的总的value值。
模型训练的目的：然个model的输出逼近在这个state能够获得的最大的所有的reward的值。

loss：利用的是MSE
"""


def train(model, epochs, verbose=1):
    """
    训练参数
    """
    epsilon = .1
    max_memory = 500
    batch_size = 1

    env = Catch(grid_size)
    exp_replay = ExperienceReplay(max_memory=max_memory)

    win_cnt = 0
    win_hist = []
    for _ in range(epochs):
        loss = 0.
        env.reset()
        game_over = False
        input_t = env.observe()  # 输入是当前游戏的状态图片

        while not game_over:
            input_tml = input_t

            # 为了防止卡在local minimum，这里加一个小的抖动，避免和之前的行为高度相关，而学习不到其他的行为，而找不到最优解
            # 这边添加的experience replay不也是这个目的吗（replay memory）？
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)[0]
            else:
                q = model.predict(input_tml)
                action = np.argmax(q[0])  # np.argmax()存在相同的最大值，返回第一个最大值位置的index

            # print(action)

            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # 如果想要可视化训练过程，取消注释
            # display_screen(action, 3000, input_t)

            exp_replay.remember([input_tml, action, reward, input_t], game_over)

            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            batch_loss = model.train_on_batch(inputs, targets)
            # print(batch_loss)

            loss += batch_loss

        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {}".format(e, epochs, loss, win_cnt))
        win_hist.append(win_cnt)

    return win_hist

if __name__ == "__main__":
    """
    模型参数
    """
    # 注意在其他地方不能够有IPython运行，也就是运行这个程序时候需要关闭jupyter notebook，如果有ipython在notebook运行的话
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()

    num_actions = 3
    hidden_size = 100
    grid_size = 10

    model = baseline_model(grid_size, num_actions, hidden_size)
    model.summary()

    # 训练过程
    # playing many games
    epoch = 5000
    hist = train(model, epoch, verbose=0)
    print("Training done")

    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
