from flask import Flask, render_template, request, jsonify
from matrix_operations import (
    matrix_addition, matrix_subtraction, matrix_multiplication,
    scalar_multiplication, determinant, inverse_matrix, transpose_matrix,
    eigenvalues_eigenvectors, matrix_rank, adjoint_matrix, matrix_trace
)
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('matrix/home.html')

@app.route('/addition', methods=['GET', 'POST'])
def addition():
    result = None
    if request.method == 'POST':
        try:
            rows_a = int(request.form['rows_a'])
            cols_a = int(request.form['cols_a'])
            matrix_a = []
            for i in range(rows_a):
                row = []
                for j in range(cols_a):
                    element = float(request.form[f'a_{i}_{j}'])
                    row.append(element)
                matrix_a.append(row)
            
            rows_b = int(request.form['rows_b'])
            cols_b = int(request.form['cols_b'])
            matrix_b = []
            for i in range(rows_b):
                row = []
                for j in range(cols_b):
                    element = float(request.form[f'b_{i}_{j}'])
                    row.append(element)
                matrix_b.append(row)
            
            result = matrix_addition(matrix_a, matrix_b)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/addition.html', result=result)

@app.route('/subtraction', methods=['GET', 'POST'])
def subtraction():
    result = None
    if request.method == 'POST':
        try:
            rows_a = int(request.form['rows_a'])
            cols_a = int(request.form['cols_a'])
            matrix_a = []
            for i in range(rows_a):
                row = []
                for j in range(cols_a):
                    element = float(request.form[f'a_{i}_{j}'])
                    row.append(element)
                matrix_a.append(row)
            
            rows_b = int(request.form['rows_b'])
            cols_b = int(request.form['cols_b'])
            matrix_b = []
            for i in range(rows_b):
                row = []
                for j in range(cols_b):
                    element = float(request.form[f'b_{i}_{j}'])
                    row.append(element)
                matrix_b.append(row)
            
            result = matrix_subtraction(matrix_a, matrix_b)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/subtraction.html', result=result)

@app.route('/multiplication', methods=['GET', 'POST'])
def multiplication():
    result = None
    if request.method == 'POST':
        try:
            rows_a = int(request.form['rows_a'])
            cols_a = int(request.form['cols_a'])
            matrix_a = []
            for i in range(rows_a):
                row = []
                for j in range(cols_a):
                    element = float(request.form[f'a_{i}_{j}'])
                    row.append(element)
                matrix_a.append(row)
            
            rows_b = int(request.form['rows_b'])
            cols_b = int(request.form['cols_b'])
            matrix_b = []
            for i in range(rows_b):
                row = []
                for j in range(cols_b):
                    element = float(request.form[f'b_{i}_{j}'])
                    row.append(element)
                matrix_b.append(row)
            
            result = matrix_multiplication(matrix_a, matrix_b)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/multiplication.html', result=result)

@app.route('/scalar_multiplication', methods=['GET', 'POST'])
def scalar_mult():
    result = None
    if request.method == 'POST':
        try:
            rows = int(request.form['rows'])
            cols = int(request.form['cols'])
            scalar = float(request.form['scalar'])
            
            matrix = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = scalar_multiplication(matrix, scalar)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/scalar_multiplication.html', result=result)

@app.route('/determinant', methods=['GET', 'POST'])
def calc_determinant():
    result = None
    if request.method == 'POST':
        try:
            size = int(request.form['size'])
            
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = determinant(matrix)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/det.html', result=result)

@app.route('/inverse', methods=['GET', 'POST'])
def calc_inverse():
    result = None
    if request.method == 'POST':
        try:
            size = int(request.form['size'])
            
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = inverse_matrix(matrix)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/inverse.html', result=result)

@app.route('/transpose', methods=['GET', 'POST'])
def calc_transpose():
    result = None
    if request.method == 'POST':
        try:
            rows = int(request.form['rows'])
            cols = int(request.form['cols'])
            
            matrix = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = transpose_matrix(matrix)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/transpose.html', result=result)

@app.route('/eigenvalues', methods=['GET', 'POST'])
def calc_eigenvalues():
    result = None
    eigenvalues = None
    eigenvectors = None
    if request.method == 'POST':
        try:
            size = int(request.form['size'])
            
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            eigenvalues, eigenvectors = eigenvalues_eigenvectors(matrix)
            result = {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/eigenvalues.html', result=result)

@app.route('/rank', methods=['GET', 'POST'])
def calc_rank():
    result = None
    if request.method == 'POST':
        try:
            rows = int(request.form['rows'])
            cols = int(request.form['cols'])
            
            matrix = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = matrix_rank(matrix)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/rank.html', result=result)

@app.route('/adjoint', methods=['GET', 'POST'])
def calc_adjoint():
    result = None
    if request.method == 'POST':
        try:
            size = int(request.form['size'])
            
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = adjoint_matrix(matrix)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/adjoint.html', result=result)

@app.route('/trace', methods=['GET', 'POST'])
def calc_trace():
    result = None
    if request.method == 'POST':
        try:
            size = int(request.form['size'])
            
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    element = float(request.form[f'matrix_{i}_{j}'])
                    row.append(element)
                matrix.append(row)
            
            result = matrix_trace(matrix)
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('matrix/trace.html', result=result)

@app.route('/generate_matrix', methods=['POST'])
def generate_matrix():
    rows = int(request.form['rows'])
    cols = int(request.form['cols'])
    matrix_id = request.form['matrix_id']
    
    return render_template('matrix/matrix_input.html', rows=rows, cols=cols, matrix_id=matrix_id)

if __name__ == '__main__':
    app.run(debug=True)