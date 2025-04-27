from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/items')
def items():
    return render_template('items.html')

@app.route('/basic')
def basic():
    return render_template('basic.html')

@app.route('/algebra')
def algebra():
    return render_template('algebra.html')

@app.route('/trigonometry')
def trigonometry():
    return render_template('trigonometry.html')

@app.route('/calculus')
def calculus():
    return render_template('calculus.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/matrices')
def matrices():
    return render_template('matrices.html')

@app.route('/characters')
def characters():
    return render_template('characters.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
