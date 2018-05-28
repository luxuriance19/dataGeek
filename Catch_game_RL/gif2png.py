"""
从gif图中截取图片
"""
import os
from PIL import Image


# im.title = [(tag,(0,0)+im.size,offset, extra)]
def analyseImage(path):
    im = Image.open(path)
    results = {'size': im.size, 'mode': "full"}
    try:
        while True:
            if im.tile:  # im.tile是2-tuple的整数
                # print(im.tile) #[('gif', (0, 0, 432, 288), 819, (8, False))]
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results["mode"] = "partial"
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def processImage(path):
    """
    迭代GIF，获得每帧
    """
    mode = analyseImage(path)["mode"]

    im = Image.open(path)

    i = 0
    p = im.getpalette()  # 返回image palette的list，也就是一个list的collor的值
    last_frame = im.convert('RGBA')

    try:
        while True:
            print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))
            '''
            如果说GIF用本地color table，每个帧会有自己的palette， 如果不是，我们需要apply global palette到新的frame
            '''
            if not im.getpalette():
                im.putpalette(p)

            new_frame = Image.new("RGBA", im.size)
            """
            如果说是"partial"mode, 这个时候每帧update的region不是原来的图片大小，所以需要构建一个在原来的图片上面的新的frame
            """
            if mode == "partial":
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert(
                "RGBA"))  # Image.paste(im, box=None, mask=None):box:4-tuple,如果是2-tuple代表左上角，如果一个图片作为第二个参数，并且没有第三个参数，box默认是(0,0)
            new_frame.save('Datageek/%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i),
                           'PNG')  # 这里去除后面的.gif

            i += 1
            last_frame = new_frame
            # 读取gif图，支持seek()和tell()方法，查找下一帧通过im.seek(im.tell()+1),或者是逆序查找到第一帧，但是不支持随机帧搜索
            # im.seek()当seek到最后一帧的时候会raise一个EOFError。
            im.seek(im.tell() + 1)

    except EOFError:
        pass


def main():
    processImage('catch_game.gif')


if __name__ == "__main__":
    main()

"""
gif图合成
"""
import imageio


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)  # duration:每一帧的间隔

    return


def main():
    image_list = ["%03d.png" % i for i in range(100)]
    gif_name = "catch_game.gif"
    create_gif(image_list, gif_name)


if __name__ == "__main__":
    main()