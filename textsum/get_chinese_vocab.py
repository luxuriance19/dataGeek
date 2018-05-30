# !/usr/bin/ python3
# coding=utf-8

import numpy as np
import re
import os
import jieba
import codecs
import collections
jieba.set_dictionary('/home/lily/Datageek/dict.txt.big')

import sys
sys.path.append('/home/lily/projects/textsummarization/textsum')

def clean_and_segmentate_data(raw_data_list, data_parent_path):
    for raw_data_file in raw_data_list:
        out_data_file = "clean." + raw_data_file
        out_data_file = data_parent_path + out_data_file
        raw_data_file = data_parent_path + raw_data_file
        # print(out_data_file, end=' ')

        # sogou encoding in GB18030
        with codecs.open(raw_data_file, 'r', encoding='GB18030') as inf:
            with open(out_data_file, 'w', encoding='utf-8') as outf:
                abstract = ""
                for i, line in enumerate(inf):
                    # every six line is a doc
                    i %= 6

                    # filter punctuation(full_width and half_width) except full_width comma(,)
                    # separate sentences using comma
                    # contenttitle and content
                    if 3 <= i <= 4:
                        # full_width brackets
                        line = re.sub(r'（.*）','',line)
                        # exclude the brackets
                        line = re.sub(r'\(.*\)','',line)
                        line = re.sub("[\s+.\!\/_,$%^*(+\"\']+|[+——！。？、~@#￥%……&*（）＂＂“”]+",'',line)

                    # contenttitle,exclude<contenttile></contenttitle>
                    if i == 3:
                        line = " ".join(jieba.cut(line[14:-15], HMM=True))
                        abstract = "abstract=<d> <p> <s> %s </s> </p> </d>" % line

                    elif i == 4:
                        line = ' '.join(jieba.cut(line[9:-10], HMM=True))

                        # filter article length < 32
                        # and long article may cause wrong
                        if len(line) > 32:
                            article = "article=<d> <p> <s> %s </s> </p> </d>" % line[:256].replace('，', '</s> <s> ')
                            temp = "publisher=AFP\t%s\t%s\n"%(abstract, article)
                            print(temp, end='')
                            outf.write(temp)
        #print(out_data_file)
        print("done")

def fullToHalf(ustring):
    rstring = ""
    for uchar in ustring:
        code = ord(uchar)
        if code == 12288: #(full-width spacekey)
            code = 32
        elif 65281 <= code <= 65374: # (ascii)
            code -= 65248
        rstring += chr(code)
    return rstring

def data_segmentation(raw_data_list, data_parent_path):
    # abstract: "abstract=<d> <p> <s> %s </s> </p> </d>"
    # article: "article=<d> <p> <s> %s </s> </p> </d>"
    # write: "publisher=AFP\t%s\t%s\n"
    for raw_data_file in raw_data_list:
        out_data_file = "segmentation." + raw_data_file
        out_data_file = data_parent_path + out_data_file
        raw_data_file = data_parent_path + raw_data_file
        print(out_data_file, end=' ')

        # sogou encoding in GB18030
        with codecs.open(raw_data_file, 'r', encoding='GB18030') as inf:
            with open(out_data_file, 'w', encoding='utf-8') as outf:
                for line in inf:
                    contenttitle = re.findall(".*<contenttitle>(.*)</contenttitle>.*", line)
                    content = re.findall(".*<content>(.*)</content>.*", line)
                    # print(contenttitle)
                    # print(content)
                    if contenttitle:
                       #  print(contenttitle)
                        summary = fullToHalf(contenttitle[0].strip()).replace(" ", ",")
                        summary = re.sub(r'\([^\)]*\)', "", summary)
                        summary = re.sub(r'[,.!?\s]+', "", summary)
                        contenttitle = " ".join(jieba.cut(summary))
                        # print(contenttitle)
                        abstract = "abstract=<d> <p> <s> %s </s> </p> </d>"%contenttitle
                    elif content:
                        # print(len(content[0]))
                        summary = fullToHalf(content[0].strip()).replace(" ", ",")
                        summary = re.sub(r"\([^\)]*\)", "", summary)
                        # print("after",summary)
                        summary = re.sub(r'[\s]+', " ", summary)
                        summary = " ".join(jieba.cut(summary))
                        # print(summary)
                        content = re.sub(r'[,，。；;.!！?？]+', "</s> <s>", summary)
                        # print(content)

                        if len(content) > 20: # some have abstract but no article
                            article = "article=<d> <p> <s> %s </s> </p> </d>" % content[:512]
                            temp = "publisher=AFP\t%s\t%s\n" % (abstract, article)
                            print(temp, end='')
                            outf.write(temp)
        print("done")


def get_chinese_vocab(data_list, data_parent_path):
    vocab = {}
    for data_file in data_list:
        data_file = data_parent_path + 'segmentation.' + data_file
        print(data_file)

    with open(data_file,'r',encoding='utf-8') as file:
        line = file.readline()
        while line:
            for word in line.split():
                try:
                    vocab[word] += 1
                except:
                    vocab[word] = 1
            line = file.readline()

    #vocab = np.array(list(vocab.items()))
    #vocab_keys = vocab[:,1].astype(int)
    #vocab = vocab[np.argsort(vocab_keys)]
    # sort dict
    vocab = sorted(vocab.items(),key = lambda x:x[1],reverse=True)

    with open(data_parent_path+'vocab1', 'w', encoding='utf-8') as f:
        for line in vocab:
            # filter frequency <= 16, chinese r'[\u4e00-\u9fa5]'
            # print(line)
            # print(line[1] > 16)
            # print(re.findall(r'[\u4e00-\u9fa5]+', line[0]))
            if line[1] >= 0 and re.findall(r'[\u4e00-\u9fa5]+', line[0]):
                print(line)
                f.write(line[0]+" "+str(line[1])+'\n')
        f.write('<s> 0\n')
        f.write('</s> 0\n')
        f.write('<p> 0\n')
        f.write('</p> 0\n')
        f.write('<d> 0\n')
        f.write('</d> 0\n')
        f.write('<UNK> 0\n')
        f.write('<PAD> 0\n')


data_segmentation(['news_tensite_xml.smarty.dat'],'/home/lily/projects/textsummarization/data/')
#clean_and_segmentate_data(['news_tensite_xml.smarty.dat'],'/home/lily/projects/textsummarization/data/')
get_chinese_vocab(['news_tensite_xml.smarty.dat'],'/home/lily/projects/textsummarization/data/')
