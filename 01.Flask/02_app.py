from flask import Flask, render_template
app = Flask(__name__) 

@app.route('/')   #이것이 정적라우팅이다.(다른 아규,파라 전달한건 아니고 위치만알려드림)
def index(): 
    return render_template('02.index.html')#기본경로templates폴더에서 시작

@app.route('/welcome')
def welcome():
    #return 'Welcome Flask!!!'                 #방법1
    return render_template('02.welcome.html') #방법2 라우팅에따라서 보여주는데이터 다른

if __name__ == '__main__': 
    app.run(debug=True) 