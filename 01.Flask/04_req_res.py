#기본복붙1
from flask import Flask, render_template, request
from flask import Response, make_response
app = Flask(__name__) 

# query parameter처리방법 request.args
@app.route('/area')
def area():
    pi = request.args.get('pi', '3.14') #만약 pi값을 사람이입력하지않으면 디폴트3.14를 주겠다(노드의 삼항같은기능)에러안나게
    radius = request.values['radius']  #1)values해도되고 args해도되고
    s = float(pi) * float(radius) * float(radius)
    return f'pi={pi}, radius={radius}, area={s}'

# form paremeter처리방법 request.form
@app.route('/login', methods=['GET','POST'])     #이것이 동적라우팅
def login():
    if request.method  == 'GET':
        return render_template('03.login.html')  #로그인폼 보여줌
    else:                                        #post면 여기실행
        uid = request.form['uid']
        pwd = request.values['pwd'] #2)values해도되고 form해도되고 (values많이사용하는건좋지않음;;)
        return f'uid={uid}, pwd={pwd}'

@app.route('/response')
def response_fn():
    custom_res = Response('Custom Response', 200, {'test': 'ttt'}) #F12>F5 Request Headers에서 test:tttt확인
    return make_response(custom_res)


#기본복붙2
if __name__ == '__main__': 
    app.run(debug=True) 