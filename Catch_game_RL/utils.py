import time
from PIL import Image
from IPython import display  # 为了render the frames 渲染frames
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


"""
定义一些variables，用可视化显示
"""
# last_frame_time： 跟踪我们到达哪个frame
last_frame_time = 0
# 将actions转成可读文字
translate_action = ["Left", "Stay", "Right", "Creat Ball", "End Test"]
# 游戏屏幕尺寸(size of the game field)
grid_size = 10


def display_screen(action, points, input_t):
    """
    用于显示game screen(render the game screen)
    """

    global last_frame_time
    # print(action)
    print("Action %s, Points: %d" % (translate_action[action], points))

    # 只在游戏没有结束的时候显示game screen
    if ("End" not in translate_action[action]):
        plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
        # 删除之前显示的图片,等到接下来的图片可以替代之前的图片的时候
        display.clear_output(wait=True)
        # 显示现在的图片
        display.display(plt.gcf())  # plt.gcf()获得现在图片的查阅

    last_frame_time = set_max_fps(last_frame_time, 5)


# 之前的函数优点问题，做了细微的修改之后可以运行，这里*1000是为了保证精度吗，
# 自己理解的这个函数的作用是：为了保证每秒最多有FPS帧画面
# FPS是frames per second
# time.time()返回浮点秒数（从1970年开始计数）
# 貌似感觉上面这种实现方法会要稳定一点
def set_max_fps(last_frame_time, FPS=1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1.0 / FPS - (current_milli_time() - last_frame_time) / 1000.
    # print(sleep_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()
    '''
    current_milli_time = time.time()
    sleep_time = 1.0/FPS - (current_milli_time - last_frame_time)
    print(sleep_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time
    '''
