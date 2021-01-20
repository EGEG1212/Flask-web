from flask import Flask, render_template, request
app = Flask(__name__) 

@app.route('/')   #이것이 정적라우팅이다.(다른 아규,파라 전달한건 아니고 위치만알려드림)
def index(): 
    return 'Hello Flask!!!'

@app.route('/user/<uid>')  #스트링타입 파라메터
def string_fn(uid):    #uid받아서
    return uid         #uid출력
@app.route('/int/<int:number>')  #<int: 이거안넣으면 5가100개나오는듯 ㅋㅋ 인트타입파라메터
def int_fn(number):    
    return str(100*number)  #str 안하면 타입에러나옴
@app.route('/float/<float:number>') #float타입 파라메터
def float_tn(number):
    return str(number * 10)
@app.route('/path/<path:path>')
def path_fn(path):
    return f'path {path}' #별다른거없이 스트링나옴;; 


@app.route('/welcome')
def welcome():
    #return 'Welcome Flask!!!'                   #방법1
    return render_template('02.welcome.html')    #방법2 라우팅에따라서 보여주는데이터 다른

@app.route('/login', methods=['GET','POST'])     #이것이 동적라우팅
def login():
    if request.method  == 'GET':
        return render_template('03.login.html')  #로그인폼 보여줌
    else:                                        #post면 여기실행
        return render_template('02.welcome.html')      #로그인프로세스실행

    

if __name__ == '__main__': 
    app.run(debug=True) 