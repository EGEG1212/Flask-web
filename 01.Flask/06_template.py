from flask import Flask ,render_template
app = Flask(__name__) 

@app.route('/hello')  
@app.route('/hello/<name>') 
def hello(name=None) :
    xdt = {'key1':'value1', 'key2':'value2'}  #dt['key1']가 dt.key1 표현가능
    return render_template('06.hello.html', name=name, dt=xdt) #아규줘야돌아감

if __name__ == '__main__':
    app.run(debug=True) 