from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai

#FOR ALGEBRA----------------------------------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
from sympy import (
    symbols, Eq, solve, factor, simplify, sympify, sqrt, root,sin, cos, tan, log, exp, Abs, I, E, pi,Pow,linsolve,
    SympifyError, parse_expr, N, Float, Rational,expand,PolynomialError,cancel,apart
)
import math
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication,convert_xor
from sympy.parsing.sympy_parser import parse_expr as sympy_parse
import re

transformations = (standard_transformations + 
                  (implicit_multiplication, convert_xor))


# Enable implicit multiplication (e.g., '2x' -> '2*x')
transformations = standard_transformations + (implicit_multiplication,)
#----------------------------------------------------------------------------------------------------------------------------------------
#FOR ALGEBRA----------------------------------------------------------------------------------------------------------------------------

from flask import Flask, render_template, request, jsonify
from matrix_operations import (
    matrix_addition, matrix_subtraction, matrix_multiplication,
    scalar_multiplication, determinant, inverse_matrix, transpose_matrix,
    eigenvalues_eigenvectors, matrix_rank, adjoint_matrix, matrix_trace
)
import numpy as np

#----------------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)

# LOAD ENV-VARIABLE FROM .env file
load_dotenv()

# GETTING API KEY
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

#LOADING MODEL
model = genai.GenerativeModel("gemini-1.5-pro")
chat = model.start_chat(history=[])


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/items')
def items():
    return render_template('items.html')

@app.route('/basic')
def basic():
    return render_template('basic.html')

'''@app.route('/algebra')
def algebra():
    return render_template('algebra.html')'''

'''@app.route('/trigonometry')
def trigonometry():
    return render_template('trigonometry.html')'''

@app.route('/calculus')
def calculus():
    return render_template('calculus.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

'''@app.route('/matrices')
def matrices():
    return render_template('matrices.html')'''

@app.route('/mensuration')
def mensuration():
    return render_template('mensuration.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    response = chat.send_message(user_message)
    bot_message = response.text
    return jsonify({'response': bot_message})

#---------------------------------------------------------- ALGEBRA ROUTES START------------------------------------------------------------------

@app.route('/algebra')
def algebra():
        return render_template('/algebra/home.html')

# 1. Simplify Expressions

def safe_parse(expr_str):
    """Safely parse a string into a SymPy expression."""
    try:
        expr = sympy_parse(expr_str, transformations=transformations)
        return expr
    except (SympifyError, SyntaxError):
        return None
    
@app.route('/algebra/simplify', methods=['GET', 'POST'])
def simplify_expr():
    result = None
    error = None
    if request.method == 'POST':
        expr_str = request.form.get('expression', '').strip()
        if not expr_str:
            error = "Please enter an expression."
        else:
            try:
                # Parse with implicit multiplication and convert ^ to **
                expr_str = expr_str.replace('^', '**')
                expr = parse_expr(expr_str, transformations=transformations)
                
                # First try to expand (for cases like (x+2)(x-3))
                expanded = expand(expr)
                
                # Then simplify (for cases like x² + 2x + 1)
                simplified = simplify(expanded)
                
                # Choose the most compact form
                result = simplified if len(str(simplified)) < len(str(expanded)) else expanded
                
            except (SympifyError, SyntaxError):
                error = "Invalid expression. Use symbols like 'x', '+', '**' for powers."
            except Exception as e:
                error = f"Processing failed: {str(e)}"
    
    return render_template('/algebra/simplify.html', result=result, error=error)

# 2. Solve Equations

transformations_ = (standard_transformations + (implicit_multiplication,))

def safe_parse_s_eq(expr_str):
    """Safely parse a mathematical expression string."""
    try:
        expr_str = expr_str.replace('^', '**')
        return parse_expr(expr_str, transformations=transformations_)
    except (SympifyError, SyntaxError):
        return None

@app.route('/algebra/solve_equation', methods=['GET', 'POST'])
def solve_eq():
    result = None
    error = None
    if request.method == 'POST':
        eq_str = request.form.get('equation', '').strip()
        if not eq_str:
            error = "Please enter an equation."
        elif '=' not in eq_str:
            error = "Equation must contain '=' sign to separate both sides."
        else:
            try:
                lhs, rhs = eq_str.split('=', 1)
                lhs_expr = safe_parse_s_eq(lhs)
                rhs_expr = safe_parse_s_eq(rhs)
                
                if lhs_expr is None or rhs_expr is None:
                    error = "Invalid equation format. Use standard operators (+, -, *, /, ^ for power)."
                else:
                    x = symbols('x')
                    eq = Eq(lhs_expr, rhs_expr)
                    solutions = solve(eq, x)
                    
                    # Format solutions properly
                    if not solutions:
                        result = "No real solutions."
                    else:
                        formatted_solutions = []
                        for sol in solutions:
                            try:
                                # Evaluate numerical solutions
                                num_val = sol.evalf()
                                if num_val.is_real:
                                    # Round to 3 decimal places
                                    rounded = round(float(num_val), 3)
                                    # Remove trailing .0 if integer
                                    if rounded == int(rounded):
                                        formatted_solutions.append(str(int(rounded)))
                                    else:
                                        formatted_solutions.append(f"{rounded:.3f}".rstrip('0').rstrip('.'))
                                else:
                                    formatted_solutions.append(str(sol))
                            except:
                                # Keep symbolic solutions as-is
                                formatted_solutions.append(str(sol))
                        
                        result = "x = " + " or x = ".join(formatted_solutions)
                        
            except Exception as e:
                error = f"Error solving equation: {str(e)}"
    
    return render_template('algebra/solve_equation.html', 
                         result=result, 
                         error=error,
                         equation_value=request.form.get('equation', ''))

# 3. Factor Polynomials
@app.route('/algebra/factor', methods=['GET', 'POST'])
def factor_poly():
    result = None
    error = None
    if request.method == 'POST':
        expr_str = request.form.get('expression', '').strip()
        
        if not expr_str:
            error = "Please enter an expression."
        else:
            try:
                # Normalize input (handle x², x^2, etc.)
                expr_str = (expr_str.replace('²', '**2')
                                   .replace('³', '**3')
                                   .replace('^', '**'))
                expr = parse_expr(expr_str, transformations=transformations)
                
                # Validate polynomial
                x = symbols('x')
                if not expr.has(x):
                    error = "Expression must contain variable 'x'"
                else:
                    # Attempt factoring with multiple methods
                    try:
                        factored = factor(expr)
                        if factored == expr:  # If couldn't factor further
                            expanded = expand(expr)
                            if expanded != expr:  # Show expanded form if different
                                result = f"Expanded form: {expanded}"
                            else:
                                result = "Expression is already in simplest form"
                        else:
                            result = f"Factored form: {factored}"
                    except PolynomialError:
                        try:
                            expanded = expand(expr)
                            result = f"Couldn't factor. Expanded form: {expanded}"
                        except:
                            error = "Unable to process this expression"
            except (SympifyError, SyntaxError) as e:
                error = f"Invalid expression format: {str(e)}"
            except Exception as e:
                error = f"Processing error: {str(e)}"
    
    return render_template('algebra/factor.html', 
                         result=str(result).replace('**', '^') if result else None,
                         error=error)

# 4. Evaluate Functions
@app.route('/algebra/function', methods=['GET', 'POST'])
def evaluate_func():
    result = None
    error = None
    if request.method == 'POST':
        expr_str = request.form.get('expression', '').strip()
        x_val = request.form.get('x', '').strip()
        
        if not expr_str or not x_val:
            error = "Please fill all fields."
        else:
            try:
                # Convert common math notations
                expr_str = (expr_str.replace('²', '**2')
                                   .replace('³', '**3')
                                   .replace('^', '**')
                                   .replace('√', 'sqrt'))
                
                # Define custom symbols
                x = symbols('x')
                local_dict = {
                    'sqrt': sqrt,
                    'sin': sin,
                    'cos': cos,
                    'tan': tan,
                    'log': log,
                    'exp': exp,
                    'abs': Abs,
                    'pi': pi,
                    'e': E
                }
                
                # Parse with custom functions
                expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
                
                # Validate
                if not expr.has(x):
                    error = "Expression must contain variable 'x'"
                else:
                    # Evaluate
                    x_num = float(x_val)
                    result = expr.subs(x, x_num)
                    
                    # Format result
                    if result.has(I):  # Complex result
                        result = str(result.evalf(4))
                    elif result.is_real:
                        result = float(result.evalf())
                        
            except ValueError:
                error = "Invalid x value. Must be a number."
            except (SympifyError, SyntaxError) as e:
                error = f"Invalid function: {str(e)}. Use valid math syntax."
            except Exception as e:
                error = f"Evaluation error: {str(e)}"
    
    return render_template('algebra/function.html',
                         result=round(result, 6) if isinstance(result, float) else result,
                         error=error)

# 5. Linear Equations (y = mx + b)
@app.route('/algebra/linear', methods=['GET', 'POST'])
def linear_eq():
    result = None
    error = None
    if request.method == 'POST':
        try:
            m = float(request.form.get('slope', 0))
            b = float(request.form.get('intercept', 0))
            x_val = float(request.form.get('x', 0))
            result = m * x_val + b
        except ValueError:
            error = "All inputs must be numbers."
    return render_template('algebra/linear.html', result=result, error=error)

# 6. Quadratic Equations (ax² + bx + c = 0)
@app.route('/algebra/quadratic', methods=['GET', 'POST'])
def quadratic_solver():
    result = None
    error = None
    
    if request.method == 'POST':
        try:
            a = float(request.form.get('a'))
            b = float(request.form.get('b'))
            c = float(request.form.get('c'))
            
            discriminant = b**2 - 4*a*c
            
            if discriminant > 0:
                root1 = (-b + math.sqrt(discriminant)) / (2*a)
                root2 = (-b - math.sqrt(discriminant)) / (2*a)
                result = [round(root1, 3), round(root2, 3)]
            elif discriminant == 0:
                root = -b / (2*a)
                result = [round(root, 3)]
            else:
                result = "No real roots (complex solution)"
                
        except ValueError:
            error = "Please enter valid coefficients"
        except ZeroDivisionError:
            error = "Coefficient 'a' cannot be zero"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('algebra/quadratic.html', result=result, error=error)

# 7. Exponents (a^b)
@app.route('/algebra/exponents', methods=['GET', 'POST'])
def exponents():
    result = None
    error = None
    if request.method == 'POST':
        base = request.form.get('base', '').strip()
        power = request.form.get('power', '').strip()
        
        if not base or not power:
            error = "Please fill both fields."
        else:
            try:
                # Parse both base and power as algebraic expressions
                x = symbols('x')
                base_expr = parse_expr(base.replace('^', '**'), transformations=transformations)
                power_expr = parse_expr(power.replace('^', '**'), transformations=transformations)
                
                # Calculate (base)^(power)
                result_expr = base_expr ** power_expr
                
                # Simplify if possible
                simplified = simplify(result_expr)
                result = str(simplified).replace('**', '^')
                
            except (SympifyError, SyntaxError):
                error = "Invalid expression. Use 'x' for variables."
            except Exception as e:
                error = f"Calculation error: {str(e)}"
    
    return render_template('algebra/exponents.html', result=result, error=error)

# 8. Radicals (n-th root)
@app.route('/algebra/radicals', methods=['GET', 'POST'])
def radicals():
    result = None
    error = None
    if request.method == 'POST':
        expr_str = request.form.get('expression', '').strip()
        root_str = request.form.get('root', '2').strip()
        
        if not expr_str:
            error = "Please enter an expression."
        else:
            try:
                # Parse the root value
                n = int(float(root_str)) if float(root_str).is_integer() else float(root_str)
                
                # Parse the expression
                x, y = symbols('x y')
                expr = parse_expr(expr_str.replace('^', '**'), transformations=transformations)
                
                # Calculate the root
                radical_expr = Pow(expr, 1/n)
                simplified = simplify(radical_expr)
                
                # Handle perfect roots and special cases
                if simplified.is_Pow and simplified.exp == 1/n:
                    # Try to factor perfect powers
                    from sympy import factor
                    factored = factor(expr)
                    if factored != expr:
                        simplified = Pow(factored, 1/n)
                        simplified = simplify(simplified)
                
                # Format the result
                if n == 2:
                    root_symbol = "√"
                elif n == 3:
                    root_symbol = "∛"
                else:
                    root_symbol = f"{n}√"
                
                # Further simplification for perfect powers
                if simplified.is_Pow and simplified.base.is_Pow:
                    base, exp = simplified.base.base, simplified.base.exp
                    new_exp = exp/n
                    if new_exp.is_integer or (1/new_exp).is_integer:
                        simplified = base**new_exp
                
                # Handle absolute value for even roots of squares
                if n == 2 and simplified.is_Pow and simplified.exp == 1 and simplified.base.is_Pow:
                    if simplified.base.exp == 2:
                        simplified = abs(simplified.base.base)
                
                # Format the output string
                input_expr = expr_str.replace('**', '^')
                output_expr = str(simplified).replace('**', '^')
                
                result = f"{root_symbol}({input_expr}) = {output_expr}"
                
            except ValueError:
                error = "Root must be a valid number"
            except (SympifyError, SyntaxError):
                error = "Invalid expression format. Use 'x' for variables and ^ for exponents."
            except Exception as e:
                error = f"Calculation error: {str(e)}"
    
    return render_template('algebra/radicals.html', 
                         result=result, 
                         error=error,
                         expression_value=request.form.get('expression', ''),
                         root_value=request.form.get('root', '2'))

@app.route('/algebra/rational', methods=['GET', 'POST'])
def rational_simplifier():
    result = None
    error = None
    if request.method == 'POST':
        numerator = request.form.get('numerator', '').strip()
        denominator = request.form.get('denominator', '').strip()
        
        if not numerator or not denominator:
            error = "Please enter both numerator and denominator."
        else:
            try:
                x = symbols('x')
                # Parse with transformations
                expr_num = parse_expr(numerator.replace('^', '**'), transformations=transformations)
                expr_den = parse_expr(denominator.replace('^', '**'), transformations=transformations)
                
                # Factor both parts
                factored_num = factor(expr_num)
                factored_den = factor(expr_den)
                
                # Simplify the fraction
                simplified = cancel((factored_num / factored_den))
                
                # Format the result properly
                if simplified.is_Pow or simplified.is_Mul or simplified.is_Add:
                    result = str(simplified).replace('**', '^')
                else:
                    result = str(simplified)
                
            except (SympifyError, SyntaxError) as e:
                error = f"Invalid expression format: {str(e)}"
            except ZeroDivisionError:
                error = "Division by zero detected in simplification"
            except Exception as e:
                error = f"Error simplifying expression: {str(e)}"
    
    return render_template('/algebra/rational.html', 
                         result=result, 
                         error=error,
                         numerator_value=request.form.get('numerator', ''),
                         denominator_value=request.form.get('denominator', ''))

# 10. system of equations
transformations = (standard_transformations + (implicit_multiplication,))

@app.route('/algebra/system_equations', methods=['GET', 'POST'])
def system_equations():
    result = None
    error = None
    input_values = {'eq1': '', 'eq2': '', 'eq3': ''}
    
    if request.method == 'POST':
        try:
            # Get and store input values
            input_values = {
                'eq1': request.form.get('eq1', '').strip(),
                'eq2': request.form.get('eq2', '').strip(),
                'eq3': request.form.get('eq3', '').strip()
            }
            
            equations = [eq for eq in input_values.values() if eq]
            
            if len(equations) < 2:
                error = "At least 2 equations are required"
            else:
                # Determine variables
                variables = symbols('x y z')[:len(equations)]
                
                # Parse equations
                parsed_eqs = []
                for eq in equations:
                    try:
                        lhs, rhs = eq.split('=', 1)
                        lhs_expr = parse_expr(lhs.replace('^', '**'), transformations=transformations)
                        rhs_expr = parse_expr(rhs.replace('^', '**'), transformations=transformations)
                        parsed_eqs.append(Eq(lhs_expr, rhs_expr))
                    except (SympifyError, ValueError, SyntaxError) as e:
                        error = f"Invalid equation format: {eq}"
                        break
                
                if not error:
                    # Solve system
                    solution = linsolve(parsed_eqs, variables)
                    
                    if not solution:
                        result = "No solution exists (system is inconsistent)"
                    else:
                        # Format solution
                        solutions = []
                        for var, val in zip(variables, solution.args[0]):
                            val_str = str(val).replace('**', '^')
                            solutions.append(f"{var} = {val_str}")
                        result = solutions
        
        except Exception as e:
            error = f"Error solving system: {str(e)}"
    
    return render_template('algebra/system_equations.html',
                         result=result,
                         error=error,
                         eq1_value=input_values['eq1'],
                         eq2_value=input_values['eq2'],
                         eq3_value=input_values['eq3'])

#---------------------------------------------------------- ALGEBRA ROUTES END------------------------------------------------------------------

from mensuration import mensuration  # <-- import mensuration blueprint
app.register_blueprint(mensuration, url_prefix='/men')

#---------------------------------------------------------- MENSURATION ROUTES END------------------------------------------------------------------


#---------------------------------------------------------- MATRIX ROUTES START------------------------------------------------------------------

@app.route('/matrices/')
def home_matrix():
    return render_template('matrix/home.html')

@app.route('/matrices/addition', methods=['GET', 'POST'])
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

@app.route('/matrices/subtraction', methods=['GET', 'POST'])
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

@app.route('/matrices/multiplication', methods=['GET', 'POST'])
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

@app.route('/matrices/scalar_multiplication', methods=['GET', 'POST'])
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

@app.route('/matrices/determinant', methods=['GET', 'POST'])
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

@app.route('/matrices/inverse', methods=['GET', 'POST'])
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

@app.route('/matrices/transpose', methods=['GET', 'POST'])
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

@app.route('/matrices/eigenvalues', methods=['GET', 'POST'])
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

@app.route('/matrices/rank', methods=['GET', 'POST'])
def calc_rank():
    result = None
    if request.method == 'POST':
        try:
            rows = int(request.form['rows'])
            cols = int(request.form['cols'])
            
            # Parse matrix from form data
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

@app.route('/matrices/adjoint', methods=['GET', 'POST'])
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

@app.route('/matrices/trace', methods=['GET', 'POST'])
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

#---------------------------------------------------------- MATRIX ROUTES END------------------------------------------------------------------


from statistics import statistics_bp
app.register_blueprint(statistics_bp, url_prefix='/statistics')

#---------------------------------------------------------- STATISTICS ROUTES END------------------------------------------------------------------

from calculus import calculus_bp
app.register_blueprint(calculus_bp, url_prefix='/calculus')

#---------------------------------------------------------- CALCULUS ROUTES END------------------------------------------------------------------

from trig import trig_bp
app.register_blueprint(trig_bp, url_prefix='/trigonometry')

if __name__ == '__main__':
    app.run(debug=True)
