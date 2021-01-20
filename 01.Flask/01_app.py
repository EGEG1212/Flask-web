from flask import Flask 
app = Flask(__name__) 

@app.route('/') 
def index(): 
    return 'by anne_ Hello Flask!!!'

if __name__ == '__main__': #여기서부터시작하면 app.run해라 코딩도장44모듈참조
    app.run(debug=True) #노드몬죽였다안살려도됨 디버그트루 럽럽