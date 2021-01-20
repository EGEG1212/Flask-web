from flask import Flask, render_template
import os       #05.index.html에서 이미지수정한거 즉각반영방법(추가1)
app = Flask(__name__) 

@app.route('/') 
def index(): 
    img_file = os.path.join(app.root_path, 'static/img/이미지.jpg') #이미지받아서 #05.index.html에서 이미지수정한거 즉각반영방법(추가2)
    mtime = int(os.stat(img_file).st_mtime)                 #모드타임구해서 #05.index.html에서 이미지수정한거 즉각반영방법(추가3)
    return render_template("05.index.html", mtime=mtime)  #아규먼트로mtime보냄


if __name__ == '__main__': 
    app.run(debug=True) 