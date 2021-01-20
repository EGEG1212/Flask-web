from flask import Blueprint, render_template, request, session, flash, redirect, url_for
from flask import current_app
from fbprophet import Prophet
from datetime import datetime, timedelta
import os
import pandas as pd
import pandas_datareader as pdr
from my_util.weather import get_weather

stock_bp = Blueprint('stock_bp', __name__)


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


nasdaq_dict, kospi_dict, kosdaq_dict = {}, {}, {}  # 기업리스트가 자주바뀌지않으니, 전역변수로만들어놓기


@stock_bp.before_app_first_request  # app안에 있어서 그런지 before_app_first_request
def before_app_first_request():
    nasdaq = pd.read_csv('./static/data/NASDAQ.csv', dtype={'Symbol': str})
    for i in nasdaq.index:
        nasdaq_dict[nasdaq['Symbol'][i]] = nasdaq['Name'][i]
    kospi = pd.read_csv('./static/data/KOSPI.csv', dtype={'종목코드': str})
    for i in kospi.index:
        kospi_dict[kospi['종목코드'][i]] = kospi['기업명'][i]
    kosdaq = pd.read_csv('./static/data/KOSDAQ.csv', dtype={'종목코드': str})
    for i in kosdaq.index:
        kosdaq_dict[kosdaq['종목코드'][i]] = kosdaq['기업명'][i]


@stock_bp.route('/stock', methods=['GET', 'POST'])
def stock():
    menu = {'ho': 0, 'da': 0, 'ml': 1, 'se': 0, 'co': 0,
            'cg': 0, 'cr': 0, 'st': 1, 'wc': 0, 're': 0}
    if request.method == 'GET':
        return render_template('/stock/stock.html', menu=menu, weather=get_weather_main(),
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

        try:
            stock_data = pdr.DataReader(
                code, data_source='yahoo', start=start_learn, end=end_learn)  # 야후주식으로부터 실데이터가져오기
            current_app.logger.debug(  # app.logger.debug에서 모듈화하면서 current_app으로변경
                f"get stock data: {code}")  # 터미널에서 확인가능한 개발자확인용
        # model에 적용시키려는 작업(눈으로 확인하려면 주피터노트북에서)
            df = pd.DataFrame({'ds': stock_data.index, 'y': stock_data.Close})
            df.reset_index(inplace=True)
            del df['Date']
        except:
            current_app.logger.error('Date error')  # 야후주식에 데이터가없으면
            flash(f'{company}_{code} 야후주식에 존재하지 않습니다', 'danger')
            return redirect(url_for('stock_bp.stock'))
        try:
            model = Prophet(daily_seasonality=True)  # 학습모델prophet에 적용시키고
            model.fit(df)
        except:
            current_app.logger.error('Value error')  # 야후주식에 데이터가없으면
            flash(f'{company}_{code} 야후주식에 존재하지 않습니다', 'danger')
            return redirect(url_for('stock_bp.stock'))
        future = model.make_future_dataframe(
            periods=pred_period)  # 사용자가 입력한 기간까지 예측한다
        forecast = model.predict(future)

        fig = model.plot(forecast)  # 그래프그리기
        img_file = os.path.join(
            current_app.root_path, 'static/img/stock.png')  # 지정한위치에 지정한파일명으로 저장해라
        fig.savefig(img_file)  # 클라이언트에게 보여주기 위해 파일저장
        mtime = int(os.stat(img_file).st_mtime)  # 이미지저장 즉각반영

        fig = model.plot_components(forecast)  # 시즌분석그래프그리기
        img_file = os.path.join(
            current_app.root_path, 'static/img/stock_plt.png')
        fig.savefig(img_file)
        mtime = int(os.stat(img_file).st_mtime)

        return render_template('/stock/stock_res.html', menu=menu, weather=get_weather_main(), mtime=mtime, company=company, code=code)
