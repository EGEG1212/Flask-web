from flask import Blueprint, render_template, request, session, flash, redirect, url_for
from flask import current_app
from werkzeug.utils import secure_filename  # 보안
from datetime import timedelta
import os
import folium
import json
import pandas as pd
from my_util.weather import get_weather
import my_util.drawKorea as dk
import my_util.drawKoreaTW as tw

carto_bp = Blueprint('carto_bp', __name__)


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


@carto_bp.route('/pop/<option>')
def population(option):
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 1, 'cr': 0, 'st': 0, 'wc': 0}
    df_pop = pd.read_csv('./static/data/population.csv')
    column_dict = {'crisis_area': '소멸위기지역', 'crisis_ratio': '소멸비율'}
    color_dict = {'crisis_area': 'Reds', 'crisis_ratio': 'PuBu'}
    img_file = os.path.join(current_app.root_path, 'static/img/population.png')
    dk.drawKorea(column_dict[option], df_pop, color_dict[option], img_file)
    mtime = int(os.stat(img_file).st_mtime)
    return render_template('cartogram/population.html', menu=menu, weather=get_weather(),
                           option=option, column_dict=column_dict, mtime=mtime)


@carto_bp.route('/coffee', methods=['GET', 'POST'])
def coffee():
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 1, 'cr': 0, 'st': 0, 'wc': 0, 're': 0}
    if request.method == 'GET':
        return render_template('cartogram/coffee.html', menu=menu, weather=get_weather())
    else:  # POST client가 업로드한 커피지수CSV 받아오기
        item = request.form['item']
        f = request.files['csv']  # coffee.html의 name="csv"부분 받아옴
        # 아래 filename = os.path.join(current_app.root_path, 'static/upload/') + secure_filename(f.filename) #static폴더아래 upload폴더생성해놓음. 파일이름은(f.filename)
        # (불특정자수중에 해커가 있을 수 있으니까)보안으로 secure_filename(f.filename) 하려면, 위에 임폴트추가from werkzeug.utils import secure_filename
        # 한글파일이름일 경우, utf8인코딩이안되는지.. log폴더안에안들어온다는문제점... logging.json파일에서 파일핸들러부분에 "encoding": "UTF-8" 추가하니 해결!
        filename = os.path.join(current_app.root_path,
                                'static/upload/') + f.filename
        f.save(filename)
        current_app.logger.info(f'{filename} is saved.')  # 개발자 확인용
        # return filename 테스트로 파일명찍혔나 확인

        coffee_index = pd.read_csv(filename, dtype={
                                   '이디야 매장수': int, '스타벅스 매장수': int, '커피빈 매장수': int, '빽다방 매장수': int})
        color_dict = {'커피지수': 'Reds', '이디야 매장수': 'Blues',
                      '스타벅스 매장수': 'Greens', '커피빈 매장수': 'Purples', '빽다방 매장수': 'PuBu'}
        img_file = os.path.join(current_app.root_path,
                                'static/img/coffee.png')  # 중요
        # 종료 후 파일생성되있겠지?
        dk.drawKorea(item, coffee_index, color_dict[item], img_file)
        mtime = int(os.stat(img_file).st_mtime)  # 그걸 받아오려고 mtime

# 그림이 심심해서 교수님이 아래 추가 탑10 ㅣby=item컬럼명 소팅
        df = coffee_index.sort_values(by=item, ascending=False)[
            ['ID', item]].reset_index()
        top10 = {}
        for i in range(10):
            top10[df['ID'][i]] = round(df[item][i], 2)
        return render_template('cartogram/coffee_res.html', menu=menu, weather=get_weather(), item=item, mtime=mtime, top10=top10)


@carto_bp.route('/coffee_view')
def coffee_view():
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 1, 'cr': 0, 'st': 0, 'wc': 0, 're': 0}
    if request.method == 'GET':
        return render_template('cartogram/coffee_view.html', menu=menu, weather=get_weather())


@carto_bp.route('/burger', methods=['GET', 'POST'])
def burger():
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'sc': 0,
            'co': 0, 'ca': 1, 'cr': 0, 'st': 0, 'wc': 0}
    if request.method == 'GET':
        return render_template('cartogram/burger.html', menu=menu, weather=get_weather())
    else:
        item = request.form['item']
        current_app.logger.debug(f"get data96: {item}")
        df_burger = pd.read_csv('./static/data/버거지수.csv', dtype={
                                '롯데리아 매장수': int, '맘스터치 매장수': int, '버거킹 매장수': int, '맥도날드매장수': int, 'KFC 매장수': int})
        img_path = current_app.root_path
        current_app.logger.debug(f"get data100: {img_path}")
        color_dict = {'버거지수': 'Blues', '롯데리아 매장수': 'Reds', '맘스터치 매장수': 'Greens',
                      '버거킹 매장수': 'PuRd', '맥도날드 매장수': 'Purples', 'KFC 매장수': 'Oranges'}
        color = color_dict[item]
        img_file = tw.drawKorea(item, df_burger, color, img_path)
        current_app.logger.debug(f"get data105: {img_file}")
        mtime = int(os.stat(img_file).st_mtime)
        df_burger_sort = df_burger.sort_values(by=[item], ascending=False)[
            ['ID', item]].reset_index().head(10)
        top10 = {}
        for i in range(10):
            top10[df_burger_sort['ID'][i]] = round(df_burger_sort[item][i], 2)
        subtitle = item
        return render_template('cartogram/burger_res.html', menu=menu, weather=get_weather(), mtime=mtime, subtitle=subtitle, top10=top10)
