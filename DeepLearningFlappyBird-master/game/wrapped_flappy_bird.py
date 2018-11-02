import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init() # 初始化所有的导入的pygame的模块，返回(numpass, numfail)不会有异常抛出，但是如果说单独的模块初始化就会有异常的抛出
FPSCLOCK = pygame.time.Clock() # 创建一个时间对象来跟踪所用的总的时间，这个对象里面也有一些方程可以帮助空值游戏的帧率
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT)) # 主要是设置在屏幕中显示额状态，这里的参数是resolution=(0,0),flags=0,depth=0(depth是说每个color存储的bits),返回的是Surface对象
pygame.display.set_caption('Flappy Bird') # 设置现在窗口的名称 参数 title, icontitle=None

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1]) # 鸟飞行的翅膀的未知，上中下中循环，为了让鸟实现动态的效果


class GameState:
    '''
    这里表示所有物件的坐标位置都是取的该物件的左上角像素点的坐标，以左上角的像素点坐标作为标准
    '''
    def __init__(self):
        '''
        self.playerx, self.playery: player的位置，playery代表的是player图片最上面的未知

        '''
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH #336-288=48

        # 这里upperPipes的参数'x'代表最左边的坐标，'y'代表
        newPipe1 = getRandomPipe() # 返回上下管道的坐标的未知
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        # 这里像素点从左上角开始是（0,0）,所以计数，向上飞是负值，向下飞是正值
        self.pipeVelX = -4         # pipeVelX: 鸟向右飞，所以管道向左移动
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

    def frame_step(self, input_actions):
        '''
        每一词动作之后的图片的状态的变化
        '''
        pygame.event.pump() # 内置进程pygame event handlers,如过不用其他的event function来让程序和操作系统的恶气它部分内置交互，那么应该要调用pygame.event.pump()让pygame空值internal actions

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        # 鸟有动作的时候，首先判断鸟所处的位置， 当鸟的合适，给鸟y轴飞行的速度
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                SOUNDS['wing'].play() # 播放音乐

        # check for score
        # playerMidPos:player的x轴的中心未知
        # pipeMidPos： 管道的中心的水平位置
        # 当鸟的中心未知超过管道的中心位置，并且超过中心位置不多与3个像素点的时候，score+1, 这个时候的reward是1
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        # loopIter 取值范围是[0,29],当loopIter每加3, PlayerIndex都发生改变
        # basex每次循环加100,取值范围为[-baseShift, 0],这边basex取非正数
        # 这里变化的目地是为了让鸟看起来有动画的效果，鸟的飞行动作不是每帧都变，而是每三帧变化一次
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        # 如果player的y轴的速度没有达到速度的最大值，并且player也没有让鸟飞，那么player的速度加上一个向下的加速度，执行这一过程之后，恢复playerFlapped为False
        # 这一部分是根据动作变更player的y轴的位置，如果说playerVelY是正值的，palyery = playery+playerVelY,这边最大的高度是BASEY嘛？为啥不是SCREENHEIGHT，这边BASEY猜测是FlappyBird的底部
        # 如果说是向上飞，那么不能超过像素0
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        # 当这个管道的已经移除屏幕的时候，将其从pipes这个list中间弹出来
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            SOUNDS['hit'].play()
            SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -1

        # draw sprites
        # .pygame.Surface.blit() 在一张图（dest）上面画另外一张图（source） blit(source,dest,area, special_flags),dest可以是表明图片的左上角的未知的一堆的坐标，也可以是一个Rect，也就是说dest实际上可以表明source放置位置的左上角
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        # pygame.surfarray.array3d()->array: 将像素从surface赋值到一个3D array里面
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # pygame.display.update()更新screen的部分显示update(rectangle=),如果说没有参数，就是更新整个Surface，类似pygame.display.flip()
        pygame.display.update()
        # pygame.time.Clock.tick: 更新clock，这个方法必须要每帧调用一次，这里会计算从上一次调用开始过去了多少ms（milliseconds），如果说带参数，类似与这里的tick(FPS)那么,就限制了每秒最大的帧数
        FPSCLOCK.tick(FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal

def getRandomPipe():
    """returns a randomly generated pipe"""
    # 这边管道中间的间隔是固定的，都是PIPEGAPSIZE
    # Pipe的的x,y都取的是左上角的像素点的坐标
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    # 显示分数在右上角
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        # pygame.Rect(left, top, width, height)返回Rect对象
        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    # rect1.clip(rect2): 返回一个新的rect，这个rect是rect1位于rect2中的部分，如果说两个rect重叠，返回0
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
