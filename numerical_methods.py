from flask import Flask, render_template, request, redirect, url_for
import math
import sympy as sp

app = Flask(__name__)

# Home page with method selection
@app.route('/')
def home():
    return render_template('home.html')

# Runge-Kutta method route
@app.route('/runge_kutta', methods=['GET', 'POST'])
def runge_kutta():
    if request.method == 'POST':
        # Get form data
        function = request.form['function']
        x0 = float(request.form['x0'])
        y0 = float(request.form['y0'])
        h = float(request.form['h'])
        target_x = float(request.form['target_x'])
        
        # Initialize variables
        steps = []
        x = x0
        y = y0
        
        # Process steps until we reach or surpass target_x
        while x < target_x:
            # Calculate Runge-Kutta coefficients
            try:
                k1 = h * eval(function.replace('x', str(x)).replace('y', str(y)))
                k2 = h * eval(function.replace('x', str(x + h/2)).replace('y', str(y + k1/2)))
                k3 = h * eval(function.replace('x', str(x + h/2)).replace('y', str(y + k2/2)))
                k4 = h * eval(function.replace('x', str(x + h)).replace('y', str(y + k3)))
            except:
                return render_template('error.html', message="Error evaluating function. Check your input.")
            
            # Calculate next y value
            delta_y = (k1 + 2*k2 + 2*k3 + k4) / 6
            y_new = y + delta_y
            
            # Record step
            steps.append({
                'x': x,
                'y': y,
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'k4': k4,
                'delta_y': delta_y,
                'y_new': y_new
            })
            
            # Update variables for next iteration
            x += h
            y = y_new
        
        return render_template('runge_kutta_result.html', 
                             function=function,
                             x0=x0,
                             y0=y0,
                             h=h,
                             target_x=target_x,
                             steps=steps,
                             final_y=y)
    
    return render_template('runge_kutta_form.html')


# Taylor method
@app.route('/taylor', methods=['GET', 'POST'])
def taylor():
    if request.method == 'POST':
        # Get form data
        function = request.form['function']
        x0 = float(request.form['x0'])
        y0 = float(request.form['y0'])
        h = float(request.form['h'])
        target_x = float(request.form['target_x'])
        order = int(request.form['order'])
        
        # Initialize variables
        steps = []
        x = x0
        y = y0
        
        # Create symbolic variables
        x_sym, y_sym = sp.symbols('x y')
        
        # Parse the function
        try:
            f = sp.sympify(function)
            f_latex = sp.latex(f)  # Get LaTeX representation of the function
        except:
            return render_template('error.html', message="Error parsing function. Check your input.")
        
        # Process steps until we reach or surpass target_x
        while x < target_x:
            step_details = {
                'x': x,
                'y': y,
                'derivatives': [],
                'terms': [],
                'term_expressions': [],  # New field for LaTeX term expressions
                'y_new': None,
                'current_expression': f"y({x:.4f}) = {y:.6f}"  # Initial expression
            }
            
            # Calculate Taylor series terms
            current_y = y
            taylor_terms = [current_y]
            current_expression = f"{y:.6f}"
            
            # Calculate derivatives
            df = f
            derivatives = []
            for i in range(1, order+1):
                # Calculate derivative
                df = sp.diff(df, x_sym) + sp.diff(df, y_sym) * f
                
                # Substitute current values
                try:
                    df_value = float(df.subs({x_sym: x, y_sym: current_y}))
                    df_latex = sp.latex(df)  # LaTeX representation of derivative
                except:
                    return render_template('error.html', message="Error calculating derivatives. Check your function.")
                
                derivatives.append(df_value)
                
                # Calculate term
                term = (df_value * (h**i)) / math.factorial(i)
                taylor_terms.append(term)
                
                # Generate LaTeX expression for the term
                term_expr = (
                    f"+ \\frac{{({h:.2f})^{{{i}}}}}{{{i}!}} \\cdot "
                    f"({df_value:.6f}) = {term:.6f}"
                )
                
                # Build current expression
                current_expression += f" + \\frac{{({h:.2f})^{{{i}}}}}{{{i}!}} \\cdot {df_value:.6f}"
                
                # Store all info for display
                step_details['derivatives'].append({
                    'order': i,
                    'expression': df_latex,
                    'value': df_value,
                    'latex': f"y^{'('*(i)}{i}{')'*(i)} = {df_latex} = {df_value:.6f}"
                })
                step_details['terms'].append({
                    'order': i,
                    'term': term,
                    'latex': term_expr
                })
                step_details['term_expressions'].append(term_expr)
            
            # Calculate new y value
            y_new = sum(taylor_terms)
            step_details['y_new'] = y_new
            step_details['current_expression'] = current_expression + f" = {y_new:.6f}"
            
            # Record step
            steps.append(step_details)
            
            # Update variables for next iteration
            x += h
            y = y_new
        
        return render_template('taylor_result.html', 
                             function=function,
                             function_latex=f_latex,
                             x0=x0,
                             y0=y0,
                             h=h,
                             target_x=target_x,
                             order=order,
                             steps=steps,
                             final_y=y)
    
    return render_template('taylor_form.html')

# Newton-Raphson method
@app.route('/newton_raphson', methods=['GET', 'POST'])
def newton_raphson():
    if request.method == 'POST':
        # Get form data
        function = request.form['function']
        x0 = float(request.form['x0'])
        tolerance = float(request.form['tolerance'])
        max_iterations = int(request.form['max_iterations'])
        
        # Initialize variables
        steps = []
        x = x0
        x_sym = sp.symbols('x')
        
        # Parse the function
        try:
            f = sp.sympify(function)
            f_prime = sp.diff(f, x_sym)
        except:
            return render_template('error.html', message="Error parsing function. Check your input.")
        
        # Perform iterations
        for i in range(max_iterations):
            try:
                f_x = float(f.subs(x_sym, x))
                f_prime_x = float(f_prime.subs(x_sym, x))
            except:
                return render_template('error.html', message="Error evaluating function or its derivative.")
            
            if abs(f_prime_x) < 1e-10:
                return render_template('error.html', message="Derivative is zero. Method failed.")
            
            x_new = x - f_x / f_prime_x
            error = abs(x_new - x)
            
            # Record step
            steps.append({
                'iteration': i + 1,
                'x': x,
                'f_x': f_x,
                'f_prime_x': f_prime_x,
                'x_new': x_new,
                'error': error
            })
            
            # Check for convergence
            if error < tolerance:
                break
                
            # Update for next iteration
            x = x_new
        
        return render_template('newton_raphson_result.html', 
                             function=function,
                             x0=x0,
                             tolerance=tolerance,
                             max_iterations=max_iterations,
                             steps=steps,
                             root=x_new,
                             converged=error < tolerance)
    
    return render_template('newton_raphson_form.html')


if __name__ == '__main__':
    app.run(debug=True)