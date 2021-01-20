import os
import folium
import json
import logging
from logging.config import dictConfig
from flask import Flask, render_template, session, escape, url_for, flash, redirect, request
from fbprophet import Prophet
from datetime import datetime, timedelta
from my_util.weather import get_weather
from my_util.forms import RegistrationForm
import pandas as pd
import pandas_datareader as pdr
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font', family='Malgun Gothic')
mpl.rc('axes', unicode_minus=False)


app = Flask(__name__)
app.secret_key = 'qwert12345'
nasdaq_dict, kospi_dict, kosdaq_dict = {}, {}, {}  # 기업리스트가 자주바뀌지않으니, 전역변수로만들어놓기

with open('./logging.json', 'r') as file:
    config = json.load(file)
dictConfig(config)
app.logger


def get_weather_main():
    weather = None
    try:
        weather = session['weather']
    except:
        app.logger.debug("get new weather info")
        weather = get_weather()
        session['weather'] = weather
        session.permanent = True
        app.permanent_session_lifetime = timedelta(minutes=60)
    return weather


@app.before_first_request  # 최초 한번만 실행하는 방식(플라스크는 이런식)
def before_first_request():
    nasdaq = pd.read_csv('./static/data/NASDAQ.csv', dtype={'Symbol': str})
    for i in nasdaq.index:
        nasdaq_dict[nasdaq['Symbol'][i]] = nasdaq['Name'][i]
    kospi = pd.read_csv('./static/data/KOSPI.csv', dtype={'종목코드': str})
    for i in kospi.index:
        kospi_dict[kospi['종목코드'][i]] = kospi['기업명'][i]
    kosdaq = pd.read_csv('./static/data/KOSDAQ.csv', dtype={'종목코드': str})
    for i in kosdaq.index:
        kosdaq_dict[kosdaq['종목코드'][i]] = kosdaq['기업명'][i]


@app.route('/')
def index():
    menu = {'ho': 1, 'da': 0, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 0, 're': 0}
    return render_template('main.html', menu=menu, weather=get_weather_main())


@app.route('/park', methods=['GET', 'POST'])
def park():
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 1,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 0, 're': 0}
    park_new = pd.read_csv('./static/data/park_info.csv')
    park_gu = pd.read_csv('./static/data/park_gu.csv')  # 적은 데이터양이라 실행될때마다 읽힌다
    park_gu.set_index('지역', inplace=True)
    if request.method == 'GET':
        map = folium.Map(location=[37.5502, 126.982], zoom_start=11)  # 맵 객체만들고
        for i in park_new.index:
            folium.CircleMarker([park_new.lat[i], park_new.lng[i]],  # 마킹하고
                                radius=int(park_new['size'][i]),
                                tooltip=f"{park_new['공원명'][i]}({int(park_new.area[i])}㎡)",
                                color='#3186cc', fill_color='#3186cc').add_to(map)
        html_file = os.path.join(app.root_path, 'static/img/park.html')
        # html_file 저장할때 map.save()해야 html이 됨(아래 stock는 fig.savefig(img_file)했었다.)
        map.save(html_file)
        mtime = int(os.stat(html_file).st_mtime)  # 즉각반영되도록
        return render_template('park.html', menu=menu, weather=get_weather_main(),
                               park_list=list(park_new['공원명']), gu_list=list(park_gu.index),
                               mtime=mtime)  # 딕셔너리 또는 리스트여야 함(시리즈데이터를 그냥 보내면 안됨, ['공원명']values하면 넘파이데이터가되니까 리스트로 감싼다)
    else:
        gubun = request.form['gubun']
        if gubun == 'park':
            park_name = request.form['name']  # 네임을읽고 아래 공원명을받아서 데이터프레임을 만든다.
            # 리셋인덱스를하는이유는 원활하게 빼내고자
            df = park_new[park_new['공원명'] == park_name].reset_index()
            park_result = {'name': park_name, 'addr': df['공원주소'][0],  # 딕셔너리만들기 이름,면적,주소,개요 자료주려고
                           'area': int(df.area[0]), 'desc': df['공원개요'][0]}
            # 폴리움지도 내용 주피터와같아요~
            map = folium.Map(location=[37.5502, 126.982], zoom_start=11)
            for i in park_new.index:
                folium.CircleMarker([park_new.lat[i], park_new.lng[i]],
                                    radius=int(park_new['size'][i]),
                                    tooltip=f"{park_new['공원명'][i]}({int(park_new.area[i])}㎡)",
                                    color='#3186cc', fill_color='#3186cc').add_to(map)
            folium.CircleMarker([df.lat[0], df.lng[0]], radius=int(df['size'][0]),
                                tooltip=f"{df['공원명'][0]}({int(df.area[0])}㎡)",
                                color='crimson', fill_color='crimson').add_to(map)  # ~요기까지 같으나, 선택된것만 crimson빨간진홍색으로
            html_file = os.path.join(
                app.root_path, 'static/img/park_res.html')  # 위치지정(map, for, folium부분을 html_file로 만듦)
            map.save(html_file)
            mtime = int(os.stat(html_file).st_mtime)
            return render_template('park_res.html', menu=menu, weather=get_weather_main(),
                                   park_result=park_result, mtime=mtime)  # 검색된 결과값 보내주고자 park_result
        else:  # 구 이름으로 검색할때
            gu_name = request.form['gu']
            # index하는이유: choropleth초로프레스로 그릴때 필요해서?
            df = park_gu[park_gu.index == gu_name].reset_index()
            # 처음하는 사람은 겸손하게.. 주피터노트북에 먼저 검토하고, 검토한 결과를 .py에 적용해야한다(어디서에러났는지 안알랴줌)
            park_result = {'gu': df['지역'][0],
                           'area': int(df['공원면적'][0]), 'm_area': int(park_gu['공원면적'].mean()),
                           # 소수점한자리까지
                           'count': df['공원수'][0], 'm_count': round(park_gu['공원수'].mean(), 1),
                           # 소수점두자리까지
                           'area_ratio': round(df['공원면적비율'][0], 2), 'm_area_ratio': round(park_gu['공원면적비율'].mean(), 2),
                           'per_person': round(df['인당공원면적'][0], 2), 'm_per_person': round(park_gu['인당공원면적'].mean(), 2)}
            # 맵 그릴때 지역명이 구네임인것만 뽑아서 지도그릴꺼라서
            df = park_new[park_new['지역'] == gu_name].reset_index()
            map = folium.Map(
                location=[df.lat.mean(), df.lng.mean()], zoom_start=13)
            for i in df.index:
                folium.CircleMarker([df.lat[i], df.lng[i]],
                                    radius=int(df['size'][i])*3,
                                    tooltip=f"{df['공원명'][i]}({int(df.area[i])}㎡)",
                                    color='#3186cc', fill_color='#3186cc').add_to(map)
            html_file = os.path.join(app.root_path, 'static/img/park_res.html')
            map.save(html_file)
            mtime = int(os.stat(html_file).st_mtime)
            return render_template('park_res2.html', menu=menu, weather=get_weather_main(),  # park_res2.html보낸다
                                   park_result=park_result, mtime=mtime)


@app.route('/park_gu/<option>')
def park_gu(option):
    menu = {'ho': 0, 'da': 1, 'ml': 0, 'se': 1,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 0}
    park_new = pd.read_csv('./static/data/park_info.csv')
    park_gu = pd.read_csv('./static/data/park_gu.csv')
    park_gu.set_index('지역', inplace=True)
    geo_str = json.load(open('./static/data/skorea_municipalities_geo_simple.json',
                             encoding='utf8'))
    map = folium.Map(location=[37.5502, 126.982],
                     zoom_start=11, tiles='Stamen Toner')
    if option == 'area':
        map.choropleth(geo_data=geo_str,
                       data=park_gu['공원면적'],
                       columns=[park_gu.index, park_gu['공원면적']],
                       fill_color='PuRd',
                       key_on='feature.id')
    elif option == 'count':
        map.choropleth(geo_data=geo_str,
                       data=park_gu['공원수'],
                       columns=[park_gu.index, park_gu['공원수']],
                       fill_color='PuRd',
                       key_on='feature.id')
    elif option == 'area_ratio':
        map.choropleth(geo_data=geo_str,
                       data=park_gu['공원면적비율'],
                       columns=[park_gu.index, park_gu['공원면적비율']],
                       fill_color='PuRd',
                       key_on='feature.id')
    elif option == 'per_person':
        map.choropleth(geo_data=geo_str,
                       data=park_gu['인당공원면적'],
                       columns=[park_gu.index, park_gu['인당공원면적']],
                       fill_color='PuRd',
                       key_on='feature.id')
    # 터미널에서 확인가능한 개발자확인용; 여기에 넣는거맞나...
    app.logger.debug(f"get option data: {option}")
    for i in park_new.index:
        folium.CircleMarker([park_new.lat[i], park_new.lng[i]],
                            radius=int(park_new['size'][i]),
                            tooltip=f"{park_new['공원명'][i]}({int(park_new.area[i])}㎡)",
                            color='green', fill_color='green').add_to(map)
    html_file = os.path.join(app.root_path, 'static/img/park_gu.html')
    map.save(html_file)
    mtime = int(os.stat(html_file).st_mtime)
    option_dict = {'area': '공원면적', 'count': '공원수',
                   'area_ratio': '공원면적 비율', 'per_person': '인당 공원면적'}
    return render_template('park_gu.html', menu=menu, weather=get_weather_main(),
                           option=option, option_dict=option_dict, mtime=mtime)


@app.route('/stock', methods=['GET', 'POST'])
def stock():
    menu = {'ho': 0, 'da': 0, 'ml': 1, 'se': 0, 'co': 0,
            'cg': 0, 'cr': 0, 'st': 1, 'wc': 0, 're': 0}
    if request.method == 'GET':
        return render_template('stock.html', menu=menu, weather=get_weather_main(),
                               nasdaq=nasdaq_dict, kospi=kospi_dict, kosdaq=kosdaq_dict)
    else:
        market = request.form['market']
        if market == 'KS':
            code = request.form['kospi_code']
            company = kospi_dict[code]
            code += '.KS'
        elif market == 'KQ':
            code = request.form['kosdaq_code']
            company = kosdaq_dict[code]
            code += '.KQ'
        else:
            code = request.form['nasdaq_code']
            company = nasdaq_dict[code]
        learn_period = int(request.form['learn'])
        pred_period = int(request.form['pred'])  # 클라이언트에게 입력받고
        today = datetime.now()  # 기준이 될 오늘날짜 받고
        start_learn = today - timedelta(days=learn_period*365)  # 기간 정리해서
        end_learn = today - timedelta(days=1)

        stock_data = pdr.DataReader(
            code, data_source='yahoo', start=start_learn, end=end_learn)  # 야후주식으로부터 실데이터가져오기
        app.logger.debug(f"get stock data: {code}")  # 터미널에서 확인가능한 개발자확인용
        # model에 적용시키려는 작업(눈으로 확인하려면 주피터노트북에서)
        df = pd.DataFrame({'ds': stock_data.index, 'y': stock_data.Close})
        df.reset_index(inplace=True)
        try:
            del df['Date']
        except:
            pass

        model = Prophet(daily_seasonality=True)  # 학습모델prophet에 적용시키고
        model.fit(df)
        future = model.make_future_dataframe(
            periods=pred_period)  # 사용자가 입력한 기간까지 예측한다
        forecast = model.predict(future)

        fig = model.plot(forecast)  # 그래프그리기
        img_file = os.path.join(
            app.root_path, 'static/img/stock.png')  # 지정한위치에 지정한파일명으로 저장해라
        fig.savefig(img_file)  # 클라이언트에게 보여주기 위해 파일저장
        mtime = int(os.stat(img_file).st_mtime)  # 이미지저장 즉각반영

        fig = model.plot_components(forecast)  # 시즌분석그래프그리기
        img_file = os.path.join(
            app.root_path, 'static/img/stock_plt.png')
        fig.savefig(img_file)
        mtime = int(os.stat(img_file).st_mtime)

        return render_template('stock_res.html', menu=menu, weather=get_weather_main(), mtime=mtime, company=company, code=code)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    menu = {'ho': 0, 'da': 0, 'ml': 0, 'se': 0, 'co': 0,
            'cg': 0, 'cr': 0, 'st': 0, 'wc': 0, 're': 1}
    if form.validate_on_submit():
        # 알람 카테고리에 따라 부트스트랩에서 다른 스타일을 적용 (success, danger)
        flash(f'{form.username.data} 님 가입 완료!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', form=form, menu=menu, weather=get_weather_main())


if __name__ == '__main__':
    app.run(debug=True)
