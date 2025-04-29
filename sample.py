from flask import Flask, render_template, request, redirect, url_for
from matrix_operations import (
    matrix_addition, matrix_subtraction, matrix_multiplication,
    scalar_multiplication, determinant, inverse_matrix, transpose_matrix,
    eigenvalues_eigenvectors, matrix_rank, adjoint_matrix, matrix_trace
)
import numpy as np

app = Flask(__name__)

@app.route('/')
def redirect_home():
    return redirect(url_for('home'))

@app.route('/matrices/')
def home():
    return render_template('matrix/home.html')

def parse_matrix_from_form(prefix="", rows_key="rows", cols_key="cols"):
    """Helper function to parse matrix from form data"""
    try:
        rows = int(request.form[rows_key])
        cols = int(request.form[cols_key])
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                element = float(request.form.get(f'{prefix}_{i}_{j}', 0))
                row.append(element)
            matrix.append(row)
        return matrix
    except Exception as e:
        raise ValueError(f"Error parsing matrix: {str(e)}")

@app.route('/matrices/addition', methods=['GET', 'POST'])
def addition():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix_a = parse_matrix_from_form(prefix="a", rows_key="rows_a", cols_key="cols_a")
            matrix_b = parse_matrix_from_form(prefix="b", rows_key="rows_b", cols_key="cols_b")
            result = matrix_addition(matrix_a, matrix_b)
        except Exception as e:
            error = str(e)
    return render_template('matrix/addition.html', result=result, error=error)

@app.route('/matrices/subtraction', methods=['GET', 'POST'])
def subtraction():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix_a = parse_matrix_from_form(prefix="a", rows_key="rows_a", cols_key="cols_a")
            matrix_b = parse_matrix_from_form(prefix="b", rows_key="rows_b", cols_key="cols_b")
            result = matrix_subtraction(matrix_a, matrix_b)
        except Exception as e:
            error = str(e)
    return render_template('matrix/subtraction.html', result=result, error=error)

@app.route('/matrices/multiplication', methods=['GET', 'POST'])
def multiplication():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix_a = parse_matrix_from_form(prefix="a", rows_key="rows_a", cols_key="cols_a")
            matrix_b = parse_matrix_from_form(prefix="b", rows_key="rows_b", cols_key="cols_b")
            result = matrix_multiplication(matrix_a, matrix_b)
        except Exception as e:
            error = str(e)
    return render_template('matrix/multiplication.html', result=result, error=error)

@app.route('/matrices/scalar_multiplication', methods=['GET', 'POST'])
def scalar_mult():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form(prefix="matrix")
            scalar = float(request.form['scalar'])
            result = scalar_multiplication(matrix, scalar)
        except Exception as e:
            error = str(e)
    return render_template('matrix/scalar_multiplication.html', result=result, error=error)

@app.route('/matrices/determinant', methods=['GET', 'POST'])
def calc_determinant():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form(rows_key="size", cols_key="size")
            result = determinant(matrix)
        except Exception as e:
            error = str(e)
    return render_template('matrix/det.html', result=result, error=error)

@app.route('/matrices/inverse', methods=['GET', 'POST'])
def calc_inverse():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form(rows_key="size", cols_key="size")
            result = inverse_matrix(matrix)
        except Exception as e:
            error = str(e)
    return render_template('matrix/inverse.html', result=result, error=error)

@app.route('/matrices/transpose', methods=['GET', 'POST'])
def calc_transpose():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form()
            result = transpose_matrix(matrix)
        except Exception as e:
            error = str(e)
    return render_template('matrix/transpose.html', result=result, error=error)

@app.route('/matrices/eigenvalues', methods=['GET', 'POST'])
def calc_eigenvalues():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form(rows_key="size", cols_key="size")
            eigenvalues, eigenvectors = eigenvalues_eigenvectors(matrix)
            result = {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors
            }
        except Exception as e:
            error = str(e)
    return render_template('matrix/eigenvalues.html', result=result, error=error)

@app.route('/matrices/rank', methods=['GET', 'POST'])
def calc_rank():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form()
            result = matrix_rank(matrix)
        except Exception as e:
            error = str(e)
    return render_template('matrix/rank.html', result=result, error=error)

@app.route('/matrices/adjoint', methods=['GET', 'POST'])
def calc_adjoint():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form(rows_key="size", cols_key="size")
            result = adjoint_matrix(matrix)
        except Exception as e:
            error = str(e)
    return render_template('matrix/adjoint.html', result=result, error=error)

@app.route('/matrices/trace', methods=['GET', 'POST'])
def calc_trace():
    error = None
    result = None
    if request.method == 'POST':
        try:
            matrix = parse_matrix_from_form(rows_key="size", cols_key="size")
            result = matrix_trace(matrix)
        except Exception as e:
            error = str(e)
    return render_template('matrix/trace.html', result=result, error=error)

@app.route('/generate_matrix', methods=['POST'])
def generate_matrix():
    rows = int(request.form.get('rows', 2))
    cols = int(request.form.get('cols', 2))
    matrix_id = request.form.get('matrix_id', 'a')

    html = f'<div class="matrix-input-container" id="{matrix_id}">'
    html += '<table class="matrix-table">'
    for i in range(rows):
        html += '<tr>'
        for j in range(cols):
            html += f'''
                <td>
                    <input type="number" step="any" name="{matrix_id}_{i}_{j}" placeholder="0" required>
                </td>
            '''
        html += '</tr>'
    html += '</table></div>'

    return html

if __name__ == '__main__':
    app.run(debug=True)