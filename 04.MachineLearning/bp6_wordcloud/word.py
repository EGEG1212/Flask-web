from flask import Blueprint, render_template, request, session, g
from flask import current_app
from werkzeug.utils import secure_filename
from datetime import timedelta
import os
import folium
import json
import pandas as pd
from my_util.weather import get_weather
from my_util.wordCloud import engCloud, hanCloud
from my_util.sports_news import get_sports_news

word_bp = Blueprint('word_bp', __name__)


def get_weather_main():
    weather = None
    try:
        weather = session['weather']
    except:
        current_app.logger.info("get new weather info")
        weather = get_weather()
        session['weather'] = weather
        session.permanent = True
        current_app.permanent_session_lifetime = timedelta(minutes=60)
    return weather


@word_bp.route('/text', methods=['GET', 'POST'])
def text():
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 1}
    if request.method == 'GET':
        return render_template('wordcloud/text.html', menu=menu, weather=get_weather())
    else:
        lang = request.form['lang']  # 한글 또는 영어 test.html에서 radio선택
        f_text = request.files['text']
        file_text = os.path.join(
            current_app.root_path, 'static/upload/') + f_text.filename
        f_text.save(file_text)
        if request.files['mask']:  # 마스크 있는경우
            f_mask = request.files['mask']
            file_mask = os.path.join(
                current_app.root_path, 'static/upload/') + f_mask.filename
            f_mask.save(file_mask)
        else:  # 마스크없는경우
            file_mask = None
        stop_words = request.form['stop_words']  # 제외어 있으나없으나?상관없음?
        current_app.logger.debug(
            f"{lang}, {f_text}, {request.files['mask']}, {stop_words}")

        text = open(file_text, encoding='utf-8').read()
        stop_words = stop_words.split(
            ' ') if stop_words else []  # 제외어 띄어쓰기로 끊어줌
        img_file = os.path.join(current_app.root_path,
                                'static/img/text.png')  # 이미지파일 저장할 경로
        img_file2 = os.path.join(current_app.root_path,
                                 'static/img/text2.png')  # 이미지파일2 저장할 경로
        if lang == 'en':  # 영어일경우, wordCloud.pydml engCloud로 처리
            engCloud(text, stop_words, file_mask, img_file)
        else:  # 한글일경우, hanCloud로 처리
            hanCloud(text, stop_words, file_mask, img_file, img_file2)

        mtime = int(os.stat(img_file).st_mtime)
        mtime = int(os.stat(img_file2).st_mtime)
        return render_template('wordcloud/text_res.html', menu=menu, weather=get_weather(),
                               filename=f_text.filename, mtime=mtime)


@word_bp.route('/sports_news')  # 버튼누르자마자 크롤링실행;;
def sports_news():
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 1}
    # 텍스트파일 저장할 경로를 미리 정해준다음
    text_file = os.path.join(current_app.root_path, 'static/data/sports.txt')
    get_sports_news(text_file)  # sport_news.py실행

    text = open(text_file, encoding='utf-8').read()
    stop_words = ['오피셜']
    mask_file = os.path.join(current_app.root_path, 'static/img/ball.jpg')
    img_file = os.path.join(current_app.root_path, 'static/img/sports.png')
    hanCloud(text, stop_words, mask_file, img_file)  # wordCloud.py실행
    mtime = int(os.stat(img_file).st_mtime)
    return render_template('wordcloud/sports_res.html', menu=menu, weather=get_weather(),
                           mtime=mtime)
