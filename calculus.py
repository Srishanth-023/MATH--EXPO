from flask import Blueprint, render_template, request

calculus_bp = Blueprint('calculus', __name__, template_folder='templates')

@calculus_bp.route('/index')
def index():
    return render_template('calculus/index.html')

@calculus_bp.route('/lhopital', methods=['GET', 'POST'])
def lhopital():
    result = None
    if request.method == 'POST':
        from sympy import symbols, limit, diff, sympify
        x = symbols('x')
        expr = sympify(request.form['expression'])
        point = sympify(request.form['point'])
        num = diff(expr, x)
        result = limit(num, x, point)
    return render_template('calculus/lhopital.html', result=result)

@calculus_bp.route('/product_rule', methods=['GET', 'POST'])
def product_rule():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, sympify
        x = symbols('x')
        f = sympify(request.form['f'])
        g = sympify(request.form['g'])
        result = diff(f * g, x)
    return render_template('calculus/product_rule.html', result=result)

@calculus_bp.route('/integration_by_parts', methods=['GET', 'POST'])
def integration_by_parts():
    result = None
    if request.method == 'POST':
        from sympy import symbols, integrate, sympify
        x = symbols('x')
        u = sympify(request.form['u'])
        dv = sympify(request.form['dv'])
        du = u.diff(x)
        v = integrate(dv, x)
        result = u * v - integrate(v * du, x)
    return render_template('calculus/integration_by_parts.html', result=result)

@calculus_bp.route('/ratio_test', methods=['GET', 'POST'])
def ratio_test():
    result = None
    if request.method == 'POST':
        from sympy import symbols, limit, Abs, simplify, sympify
        n = symbols('n')
        a_n = sympify(request.form['sequence'])
        ratio = simplify(a_n.subs(n, n+1) / a_n)
        result = limit(Abs(ratio), n, float('inf'))
    return render_template('calculus/ratio_test.html', result=result)

@calculus_bp.route('/partial_derivatives', methods=['GET', 'POST'])
def partial_derivatives():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, sympify
        x, y = symbols('x y')
        expr = sympify(request.form['expression'])
        partial_x = diff(expr, x)
        partial_y = diff(expr, y)
        result = (partial_x, partial_y)
    return render_template('calculus/partial_derivatives.html', result=result)

@calculus_bp.route('/separation_of_variables', methods=['GET', 'POST'])
def separation_of_variables():
    result = None
    if request.method == 'POST':
        from sympy import symbols, Function, Eq, dsolve, sympify
        x = symbols('x')
        y = Function('y')
        expr = sympify(request.form['expression'])
        ode = Eq(y(x).diff(x), expr)
        result = dsolve(ode, y(x))
    return render_template('calculus/separation_of_variables.html', result=result)

@calculus_bp.route('/optimization', methods=['GET', 'POST'])
def optimization():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, solve, sympify
        x = symbols('x')
        func = sympify(request.form['function'])
        first_derivative = diff(func, x)
        critical_points = solve(first_derivative, x)
        second_derivative = diff(first_derivative, x)
        result = {
            'critical_points': critical_points,
            'second_derivative': second_derivative
        }
    return render_template('calculus/optimization.html', result=result)

@calculus_bp.route('/implicit_differentiation', methods=['GET', 'POST'])
def implicit_differentiation():
    result = None
    if request.method == 'POST':
        from sympy import symbols, Function, Eq, diff, dsolve, solve, sympify
        x = symbols('x')
        y = Function('y')
        expr = sympify(request.form['expression'])
        equation = Eq(expr, 0)
        dy_dx = solve(diff(equation.lhs, x) + diff(equation.lhs, y(x)) * y(x).diff(x), y(x).diff(x))
        result = dy_dx
    return render_template('calculus/implicit_differentiation.html', result=result)

@calculus_bp.route('/related_rates', methods=['GET', 'POST'])
def related_rates():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, sympify
        x, t = symbols('x t')
        expr = sympify(request.form['expression'])
        dx_dt = diff(expr, t)
        result = dx_dt
    return render_template('calculus/related_rates.html', result=result)

@calculus_bp.route('/maclaurin_series', methods=['GET', 'POST'])
def maclaurin_series():
    result = None
    if request.method == 'POST':
        from sympy import symbols, series, sympify
        x = symbols('x')
        func = sympify(request.form['function'])
        n = int(request.form['order'])
        result = series(func, x, 0, n+1).removeO()
    return render_template('calculus/maclaurin_series.html', result=result)

@calculus_bp.route('/directional_derivatives', methods=['GET', 'POST'])
def directional_derivatives():
    result = None
    if request.method == 'POST':
        from sympy import symbols, Matrix, diff, sympify
        x, y = symbols('x y')
        f = sympify(request.form['function'])
        grad_f = Matrix([diff(f, x), diff(f, y)])
        direction = Matrix(eval(request.form['direction']))
        unit_direction = direction / direction.norm()
        result = grad_f.dot(unit_direction)
    return render_template('calculus/directional_derivatives.html', result=result)

@calculus_bp.route('/line_integrals', methods=['GET', 'POST'])
def line_integrals():
    result = None
    if request.method == 'POST':
        from sympy import symbols, integrate, sympify
        t = symbols('t')
        f = sympify(request.form['function'])
        x = sympify(request.form['x(t)'])
        y = sympify(request.form['y(t)'])
        dx = x.diff(t)
        dy = y.diff(t)
        integrand = f.subs({'x': x, 'y': y}) * (dx**2 + dy**2)**0.5
        result = integrate(integrand, (t, float(request.form['t0']), float(request.form['t1'])))
    return render_template('calculus/line_integrals.html', result=result)

@calculus_bp.route('/surface_integrals', methods=['GET', 'POST'])
def surface_integrals():
    result = None
    if request.method == 'POST':
        from sympy import symbols, integrate, sympify
        x, y = symbols('x y')
        f = sympify(request.form['function'])
        region = (x, float(request.form['x0']), float(request.form['x1'])), (y, float(request.form['y0']), float(request.form['y1']))
        result = integrate(f, *region)
    return render_template('calculus/surface_integrals.html', result=result)

@calculus_bp.route('/vector_calculus', methods=['GET', 'POST'])
def vector_calculus():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, sympify
        x, y, z = symbols('x y z')
        F = [sympify(request.form['Fx']), sympify(request.form['Fy']), sympify(request.form['Fz'])]
        div_F = diff(F[0], x) + diff(F[1], y) + diff(F[2], z)
        result = div_F
    return render_template('calculus/vector_calculus.html', result=result)

@calculus_bp.route('/greens_theorem', methods=['GET', 'POST'])
def greens_theorem():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, integrate, sympify
        x, y = symbols('x y')
        P = sympify(request.form['P'])
        Q = sympify(request.form['Q'])
        curl = diff(Q, x) - diff(P, y)
        region = (x, float(request.form['x0']), float(request.form['x1'])), (y, float(request.form['y0']), float(request.form['y1']))
        result = integrate(curl, *region)
    return render_template('calculus/greens_theorem.html', result=result)

@calculus_bp.route('/stokes_theorem', methods=['GET', 'POST'])
def stokes_theorem():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, Matrix, sympify
        x, y, z = symbols('x y z')
        F = Matrix([sympify(request.form['Fx']), sympify(request.form['Fy']), sympify(request.form['Fz'])])
        curl = F.cross(Matrix([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]))
        result = curl
    return render_template('calculus/stokes_theorem.html', result=result)

@calculus_bp.route('/improper_integrals', methods=['GET', 'POST'])
def improper_integrals():
    result = None
    if request.method == 'POST':
        from sympy import symbols, integrate, sympify, oo
        x = symbols('x')
        f = sympify(request.form['function'])
        result = integrate(f, (x, float(request.form['lower_limit']), oo))
    return render_template('calculus/improper_integrals.html', result=result)

@calculus_bp.route('/partial_derivatives_holding_constant', methods=['GET', 'POST'])
def partial_derivatives_holding_constant():
    result = None
    if request.method == 'POST':
        from sympy import symbols, diff, sympify
        x, y, z = symbols('x y z')
        f = sympify(request.form['expression'])
        partial = diff(f, x)
        result = partial
    return render_template('calculus/partial_derivatives_holding_constant.html', result=result)

@calculus_bp.route('/jacobian_determinant', methods=['GET', 'POST'])
def jacobian_determinant():
    result = None
    if request.method == 'POST':
        from sympy import symbols, Matrix, diff, sympify
        x, y = symbols('x y')
        u = sympify(request.form['u'])
        v = sympify(request.form['v'])
        J = Matrix([[diff(u, x), diff(u, y)], [diff(v, x), diff(v, y)]])
        result = J.det()
    return render_template('calculus/jacobian_determinant.html', result=result)

@calculus_bp.route('/center_of_mass', methods=['GET', 'POST'])
def center_of_mass():
    result = None
    if request.method == 'POST':
        from sympy import symbols, integrate, sympify
        x, y = symbols('x y')
        rho = sympify(request.form['density'])
        region = (x, float(request.form['x0']), float(request.form['x1'])), (y, float(request.form['y0']), float(request.form['y1']))
        mass = integrate(rho, *region)
        x_bar = integrate(x * rho, *region) / mass
        y_bar = integrate(y * rho, *region) / mass
        result = (x_bar, y_bar)
    return render_template('calculus/center_of_mass.html', result=result)