import numpy as np
from collections import deque # 双端队列

# 设置游戏过程：
class Catch(object):
    """
    class Catch是真实的游戏过程
    In the game, white tiles(白色的砖块) 从顶部掉落
    goal：通过basket（用white tiles代替）来接fruits。
    action: left:0, stay: 1, right: 2
    """

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        state: f0,f1,basket
            f0,f1: 表示砖块下落的位置
            basktet： 表示basket的位置
        输入： states and actions
        输出： new states and reward
        """

        state = self.state
        if action == 0:  # 0: left
            action = -1
        elif action == 1:  # 1: stay
            action = 0
        else:
            action = 1

        f0, f1, basket = state[0]
        # 接砖块的篮子必须放置在grid_size的范围之内，这边要求的范围是[1, self.grid_size-1]
        new_basket = min(max(1, basket + action), self.grid_size - 1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]  # 等同于 out[np.newaxis,:]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        """
        给出游戏界面
        """
        im_size = (self.grid_size,) * 2
        state = self.state[0]
        canvas = np.zeros(im_size)
        # 画下落的白色砖块的位置
        canvas[state[0], state[1]] = 1
        # 画下面接白色砖块的篮子的位置,在图片的底部，篮子的宽度是3个white tile的宽度
        canvas[-1, state[2] - 1: state[2] + 2] = 1
        return canvas

    def _get_reward(self):
        """
        回馈reward
        """
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size - 1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size - 1, size=1)  # np.random.randint(low,high),[low,high)
        m = np.random.randint(1, self.grid_size - 2, size=1)  # 这里只能够是[1,grid_size-2)的原因是因为basket的长度是3
        self.state = np.asarray([0, n, m])[np.newaxis]


"""
NN的输入：<s,a>
NN的输出：Q(s,a)

training process:
experineces: <s,a,r,s'>(s:current state, a:current action, r: instant reward, s': following state)
1. 对每个action计算Q(s',a')(Q值，也就是与state和action都有关的value function)
2. 选取这三个不同action的最大的Q值
3. 计算带有discount factor（衰减因子）:gamma的总的Q值：Q(s,a) = r + gamma*max(Q(s',a'))。这个就是神经网络的目标值
4. 利用loss function: 1//2*(predict_Q(s,a)-target)^2

所有的experience放置在replay memory当中
"""


class ExperienceReplay():  # Python3自动继承object类，所以我后面基本上都省略了
    """
    在gameplay时期的所有的experience<s,a,r,s'>都自动存储在replay memory当中,这里memory是列表
    在训练的时候，在replay memory中的随机抽样batches的experiences作为input和output用来训练
    """

    def __init__(self, max_memory=100, discount=.9):
        """
        max_memory: 可以存储的experiences的最大长度
        memory：list of experiences,元素是[experience, game_over(bool)] ,list of list,experience有四个元素
        experience： [游戏初始界面，action, reward, 随后的游戏界面]
        discount: discount factor
        """
        # ============================================================================
        # self.max_memory = max_memory
        # self.memory = []
        # ============================================================================
        self.memory = deque(maxlen=max_memory)
        self.discount = discount

    def remember(self, states, game_over):
        # 将state存储在memory当中
        self.memory.append([states, game_over])

        # ==============================================================================
        # 这里list是可变长的，所以，需要删除最大长度的，为啥我在这里不用collecttion.queue()来实现呢
        # 简要查了一下相关的，暂时没有找到详细的list和queue的实现，没有比较他们的计算复杂度
        # if len(self.memory) > self.max_memory:
        #     del self.memory[0]
        # ==============================================================================

    def get_batch(self, model, batch_size=10):
        """
        len_memory: 存储了多少experience
        num_actions: 计算在game中有多少中actions可以被采取
        env_dim: game field的维度
        inputs：batches of inputs([cur_state,action,reward, follwed_state])
        targets: batches of 目标函数
        """
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        # print(self.memory)
        # memory的构成是list of experience,在Catch.obeserve()返回当前图片的时候已经将其转换成一维的
        # 向量，所以这里实际上就是grid_size**2
        env_dim = self.memory[0][0][0].shape[1]
        # print(env_dim)
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            """
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tpl: the state that followed s'
            """
            state_t, action_t, reward_t, state_tpl = self.memory[idx][0]
            game_over = self.memory[idx][1]
            # inputs[i:i+1] = state_t
            inputs[i] = state_t
            # print(model.predict(state_t).shape)
            targets[i] = model.predict(state_t)[0]  # model.predict(state_t) 输出的维度是[1，3]

            Q_sa = np.max(model.predict(state_tpl)[0])

            if game_over:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets