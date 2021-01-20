import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
import re
from konlpy.tag import Okt


def engCloud(text, stop_words, mask_file, img_file, max_words=1000):
    stopwords = set(STOPWORDS)
    for sw in stop_words:
        stopwords.add(sw)

    if mask_file == None:  # 마스크없을경우
        wc = WordCloud(background_color='black', width=800,
                       height=800, max_words=max_words, stopwords=stopwords)
    else:  # 마스크있는경우
        mask = np.array(Image.open(mask_file))
        wc = WordCloud(background_color='white', width=800, height=800,
                       max_words=max_words, mask=mask, stopwords=stopwords)
    wc = wc.generate(text)  # 영어분석은 한줄이면 분석 끝;
    plt.figure(figsize=(8, 8), dpi=100)  # dpi 화면꽉차게
    ax = plt.axes([0, 0, 1, 1])
    #plt.imshow(wc, interpolation='bilinear')
    plt.imshow(wc, interpolation='nearest', aspect='equal')
    plt.axis('off')
    plt.savefig(img_file)


def hanCloud(text, stop_words, mask_file, img_file, img_file2, max_words=1000):
    mpl.rc('font', family='Malgun Gothic')  # plt그래프에 한글폰트사용하기 위해 필요한 두줄
    mpl.rc('axes', unicode_minus=False)
    okt = Okt()  # konlpy사용 04_Nacer_KIN참고
    tokens = okt.nouns(text)  # 명사추출. 다소시간소요됨
    new_text = []  # 여기부터 영문자,숫자 제거
    for token in tokens:
        text = re.sub('[a-zA-Z0-9]', '', token)
        new_text.append(text)  # 여기까지
    new_text = [word for word in new_text if word not in stop_words]  # 제외단어 제거
    han_text = nltk.Text(new_text, name='한글 텍스트')  # 단어제거후 다시확인nltk.Text
    data = han_text.vocab().most_common(300)  # 빈도수높은 300단어
    if mask_file == None:  # 마스크파일이없다면
        wc = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
                       width=800, height=800,
                       relative_scaling=0.2, background_color='black',
                       ).generate_from_frequencies(dict(data))
    else:  # 마스크파일이 있다면
        mask = np.array(Image.open(mask_file))
        wc = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
                       width=800, height=800,
                       relative_scaling=0.2, mask=mask,
                       background_color='white',
                       ).generate_from_frequencies(dict(data))

# 마스크에얹은이미지s
    plt.figure(figsize=(5, 5), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    #plt.imshow(wc, interpolation='bilinear')
    plt.imshow(wc, interpolation='nearest', aspect='equal')
    plt.axis('off')
    plt.savefig(img_file)

    # 단어제거 후 확인그래프 추가
    htext = nltk.Text(han_text, name='한글 텍스트')
    plt.figure(figsize=(5, 5), dpi=100)
    htext.plot(15)  # 30단어 그래프
    plt.savefig(img_file2)
