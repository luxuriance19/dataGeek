import pygame
import sys
def load():
    """
    存储类型{key:value(tuple)}
    IMAGES: 所有的游戏的背景的图片
    SOUNDS： 游戏的背景声音
    HITMASKS： 如果说=0,说名没有撞到管道，每一个图片是用嵌套list存储的
    """
    # path of player with different states
    # 这里载入了小鸟飞行时候的三种不同的状态，一个图片代表一个飞行的状态
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap.png', # 向上飞行
            'assets/sprites/redbird-midflap.png', # 水平飞行
            'assets/sprites/redbird-downflap.png' # 向下飞行
    )

    # path of background
    # 背景颜色都是黑色的
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # path of pipe
    # 需要越过的管道
    PIPE_PATH = 'assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    # 这里是分数显示的图片，通过pygame.image.load()导入
    # pygame.image.load()的参数可以是filename，也可以是Python file-like对象，Pygame会自动的决定image的type（例如GIF或者bitmap），并且从data中创建一个新的Surface object，
    # 如过说传递的是一个file-like对象，那么序奥另外一个参数load(fileobj,namehint="")来高素原来的图片是什么格式的，也就是需要标明后缀。
    # 对于这个load()返回的的对象会包含和这个file相同的color format，colorkey还有alpha透明度。添加Surface.convert()不需要任何参数，可以创建一个复制版本，让图片在screen上面更新更快
    # 对于alpha transparency（这里认为是图片的透明度），类似与在.png images，使用convert_alpha()让图片有per pixel transparency
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    # 以下代码查看是哪个运行平台
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    # pygame.mixer.Sound()可以读入file或者是buffer object创建一个新的声音对象，
    # 关键字是pygame.mixer.Sound(file=filename)/buffer=buffer 官方文档说加关键字不容以弄混，这边提到的audio file只说的ogg和wav后缀的文件
    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    # pygame.transform.rotate(Surface, angle)->Surface
    # 上下两根管子
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    # image.get_width()返回的是image的宽的像素值
    # image.get_at((x,y)) 获得一个单独的像素点的颜色的值，返回的是RGBA的值(red, green, blue, alpha),如果说没有pixel alpha，那么alpha的值就之一是255
    # mask就是alpha的值
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
