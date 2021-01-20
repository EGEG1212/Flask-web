from my_util.forms import RegistrationForm
from flask import Flask, render_template, session, request, g, redirect, flash, url_for
from datetime import timedelta
import os
import json
import logging
from logging.config import dictConfig
from bp1_seoul.seoul import seoul_bp
from bp2_covid.covid import covid_bp
from bp3_cartogram.carto import carto_bp
from bp4_crawling.crawl import crawl_bp
from bp5_wordcloud.word import word_bp
from bp6_classification.clsf import clsf_bp
from bp7_advanced.aclsf import aclsf_bp
from bp8_regression.rgrs import rgrs_bp
from my_util.weather import get_weather

app = Flask(__name__)
app.secret_key = 'qwert12345'
app.config['SESSION_COOKIE_PATH'] = '/'
app.register_blueprint(seoul_bp, url_prefix='/seoul')
app.register_blueprint(covid_bp, url_prefix='/covid')
app.register_blueprint(carto_bp, url_prefix='/cartogram')
app.register_blueprint(crawl_bp, url_prefix='/crawling')
app.register_blueprint(word_bp, url_prefix='/wordcloud')
app.register_blueprint(clsf_bp, url_prefix='/classification')
app.register_blueprint(aclsf_bp, url_prefix='/advanced')
app.register_blueprint(rgrs_bp, url_prefix='/regression')

with open('./logging.json', 'r') as file:
    config = json.load(file)
dictConfig(config)
# app.logger


def get_weather_main():
    # weather = None
    # try:
    #     weather = session['weather']
    # except:
    #     app.logger.debug("get new weather info")
    #     weather = get_weather()
    #     session['weather'] = weather
    #     session.permanent = True
    #     app.permanent_session_lifetime = timedelta(minutes=1)
    weather = get_weather()
    return weather


@app.route('/')
def index():
    menu = {'ho': 1, 'da': 0, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 0, 'cf': 0, 'ac': 0, 're': 0, 'cu': 0, 're': 0}
    return render_template('index.html', menu=menu, weather=get_weather())


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    menu = {'ho': 1, 'da': 0, 'ml': 0, 'se': 0,
            'co': 0, 'cg': 0, 'cr': 0, 'st': 0, 'wc': 0, 'cf': 0, 'ac': 0, 're': 0, 'cu': 0, 're': 0}
    if form.validate_on_submit():
        # 알람 카테고리에 따라 부트스트랩에서 다른 스타일을 적용 (success, danger)
        flash(f'{form.username.data} 님 가입 완료!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', form=form, menu=menu, weather=get_weather())


if __name__ == '__main__':
    app.run(debug=True)
