from flask import Blueprint, render_template, request, jsonify
import math
import cmath

trig_bp = Blueprint('trig', __name__, template_folder='templates')

# Helper Functions
def validate_triangle_angles(angle_a, angle_b, angle_c):
    """Validate that angles sum to 180 degrees with floating point tolerance"""
    if angle_a and angle_b and angle_c:
        return abs(angle_a + angle_b + angle_c - 180) < 0.001
    return True

def get_exact_values(angle):
    """Return exact values for special angles (0°, 30°, 45°, 60°, 90°)"""
    exact = {
        'sin': {0: '0', 30: '1/2', 45: '√2/2', 60: '√3/2', 90: '1'},
        'cos': {0: '1', 30: '√3/2', 45: '√2/2', 60: '1/2', 90: '0'},
        'tan': {0: '0', 30: '√3/3', 45: '1', 60: '√3', 90: '∞'}
    }
    return {fn: exact[fn][angle] for fn in ['sin', 'cos', 'tan']}

def get_inverse_range(function):
    """Return the principal value range for inverse trig functions"""
    ranges = {
        'asin': "-90° to 90°",
        'acos': "0° to 180°",
        'atan': "-90° to 90°"
    }
    return ranges.get(function, "")

# 1. Trigonometry Home Page
@trig_bp.route('/home')
def home():
    return render_template('trig/home.html')

# 2. Right Triangle Solver
@trig_bp.route('/right-triangle', methods=['GET', 'POST'])
def right_triangle():
    if request.method == 'POST':
        data = request.json
        side_a = float(data.get('a', 0)) if data.get('a') else 0
        side_b = float(data.get('b', 0)) if data.get('b') else 0
        side_c = float(data.get('c', 0)) if data.get('c') else 0
        angle_a = float(data.get('A', 0)) if data.get('A') else 0
        angle_b = float(data.get('B', 0)) if data.get('B') else 0
        
        # Calculation logic for right triangles
        if side_a and side_b:
            side_c = math.hypot(side_a, side_b)
            angle_a = math.degrees(math.atan(side_a/side_b))
            angle_b = 90 - angle_a
        elif side_a and side_c:
            side_b = math.sqrt(side_c**2 - side_a**2)
            angle_a = math.degrees(math.asin(side_a/side_c))
            angle_b = 90 - angle_a
        elif side_b and side_c:
            side_a = math.sqrt(side_c**2 - side_b**2)
            angle_b = math.degrees(math.asin(side_b/side_c))
            angle_a = 90 - angle_b
        elif side_a and angle_a:
            side_b = side_a / math.tan(math.radians(angle_a))
            side_c = side_a / math.sin(math.radians(angle_a))
            angle_b = 90 - angle_a
        elif side_b and angle_a:
            side_a = side_b * math.tan(math.radians(angle_a))
            side_c = side_b / math.cos(math.radians(angle_a))
            angle_b = 90 - angle_a
        elif side_c and angle_a:
            side_a = side_c * math.sin(math.radians(angle_a))
            side_b = side_c * math.cos(math.radians(angle_a))
            angle_b = 90 - angle_a
        
        return jsonify({
            'a': round(side_a, 4) if side_a else None,
            'b': round(side_b, 4) if side_b else None,
            'c': round(side_c, 4) if side_c else None,
            'A': round(angle_a, 4) if angle_a else None,
            'B': round(angle_b, 4) if angle_b else None
        })
    return render_template('trig/right_triangle.html')

# 3. Basic Trigonometric Ratios
@trig_bp.route('/basic-ratios', methods=['GET', 'POST'])
def basic_ratios():
    if request.method == 'POST':
        data = request.json
        angle = float(data['angle'])
        radians = math.radians(angle)
        
        return jsonify({
            'sin': round(math.sin(radians), 4),
            'cos': round(math.cos(radians), 4),
            'tan': round(math.tan(radians), 4) if abs(math.tan(radians)) < 1e10 else "undefined"
        })
    return render_template('trig/basic_ratios.html')

# 4. Reciprocal Trigonometric Ratios
@trig_bp.route('/reciprocal-ratios', methods=['GET', 'POST'])
def reciprocal_ratios():
    if request.method == 'POST':
        data = request.json
        angle = float(data['angle'])
        radians = math.radians(angle)
        
        sin_val = math.sin(radians)
        cos_val = math.cos(radians)
        tan_val = math.tan(radians)
        
        return jsonify({
            'csc': round(1/sin_val, 4) if sin_val != 0 else "undefined",
            'sec': round(1/cos_val, 4) if cos_val != 0 else "undefined",
            'cot': round(1/tan_val, 4) if tan_val != 0 else "undefined"
        })
    return render_template('trig/reciprocal_ratios.html')

# 5. Special Angles
@trig_bp.route('/special-angles', methods=['GET', 'POST'])
def special_angles():
    if request.method == 'POST':
        data = request.json
        angle = float(data['angle'])
        
        if angle not in [0, 30, 45, 60, 90]:
            return jsonify({'error': 'Angle must be 0°, 30°, 45°, 60°, or 90°'})
        
        radians = math.radians(angle)
        return jsonify({
            'angle': angle,
            'sin': round(math.sin(radians), 4),
            'cos': round(math.cos(radians), 4),
            'tan': round(math.tan(radians), 4) if abs(math.tan(radians)) < 1e10 else "undefined",
            'exact': get_exact_values(angle)
        })
    return render_template('trig/special_angles.html')

# 6. Complementary Angle Identities
@trig_bp.route('/complementary_angles', methods=['GET', 'POST'])
def complementary_angles():
    if request.method == 'POST':
        data = request.json
        angle = float(data['angle'])
        
        if angle > 90:
            return jsonify({'error': 'Angle must be ≤ 90°'})
        
        comp_angle = 90 - angle
        sin_val = math.sin(math.radians(angle))
        cos_comp = math.cos(math.radians(comp_angle))
        
        return jsonify({
            'angle': angle,
            'complement': comp_angle,
            'sin(θ)': round(sin_val, 4),
            'cos(90°-θ)': round(cos_comp, 4),
            'equal': round(sin_val, 4) == round(cos_comp, 4)
        })
    return render_template('complementary_angles.html')

# 7. Unit Circle
@trig_bp.route('/unit_circle', methods=['GET', 'POST'])
def unit_circle():
    if request.method == 'POST':
        data = request.json
        angle = float(data['angle'])
        radians = math.radians(angle)
        
        x = math.cos(radians)
        y = math.sin(radians)
        
        return jsonify({
            'angle': angle,
            'x': round(x, 4),
            'y': round(y, 4),
            'sin': round(y, 4),
            'cos': round(x, 4),
            'tan': round(y/x, 4) if x != 0 else "undefined"
        })
    return render_template('unit_circle.html')

# 8. Graphs of Trigonometric Functions
@trig_bp.route('/trig_graphs', methods=['GET', 'POST'])
def trig_graphs():
    if request.method == 'POST':
        data = request.json
        function = data['function']
        amplitude = float(data.get('amplitude', 1))
        period = float(data.get('period', 360))
        phase_shift = float(data.get('phase_shift', 0))
        vertical_shift = float(data.get('vertical_shift', 0))
        
        # Generate points for the graph
        points = []
        for x in range(0, 721, 15):  # 0° to 720° in 15° increments
            radians = math.radians(x - phase_shift)
            scaled_x = x * (360 / period)
            
            if function == 'sin':
                y = amplitude * math.sin(math.radians(scaled_x - phase_shift)) + vertical_shift
            elif function == 'cos':
                y = amplitude * math.cos(math.radians(scaled_x - phase_shift)) + vertical_shift
            elif function == 'tan':
                try:
                    y = amplitude * math.tan(math.radians(scaled_x - phase_shift)) + vertical_shift
                    if abs(y) > 10:  # Limit extreme values for better visualization
                        y = None
                except:
                    y = None
            
            points.append({'x': x, 'y': round(y, 4) if y is not None else None})
        
        return jsonify({'points': points})
    return render_template('trig_graphs.html')

# 9. Amplitude, Period, and Phase Shift
@trig_bp.route('/amplitude_period_phase', methods=['GET', 'POST'])
def amplitude_period_phase():
    if request.method == 'POST':
        data = request.json
        a = float(data['amplitude'])
        b = float(data['period_coeff'])
        c = float(data['phase_shift'])
        d = float(data['vertical_shift'])
        
        period = 360 / abs(b)
        phase_shift = -c / b
        
        return jsonify({
            'amplitude': a,
            'period': round(period, 2),
            'phase_shift': round(phase_shift, 2),
            'vertical_shift': d,
            'equation': f"y = {a} * trig({b}(x {'+' if c >=0 else ''}{c})) {'+' if d >=0 else ''}{d}"
        })
    return render_template('amplitude_period_phase.html')

# 10. Solving Trigonometric Equations
@trig_bp.route('/solve_trig_equations', methods=['GET', 'POST'])
def solve_trig_equations():
    if request.method == 'POST':
        data = request.json
        equation_type = data['type']
        value = float(data['value'])
        
        solutions = []
        if equation_type == 'sin':
            principal = math.degrees(math.asin(value))
            solutions = [principal, 180 - principal]
        elif equation_type == 'cos':
            principal = math.degrees(math.acos(value))
            solutions = [principal, 360 - principal]
        elif equation_type == 'tan':
            principal = math.degrees(math.atan(value))
            solutions = [principal]
        
        # Generate general solutions
        general = []
        for sol in solutions:
            if equation_type == 'sin':
                general.append(f"{round(sol, 2)}° + 360°n")
                general.append(f"{round(180 - sol, 2)}° + 360°n")
            elif equation_type == 'cos':
                general.append(f"{round(sol, 2)}° + 360°n")
                general.append(f"{round(-sol, 2)}° + 360°n")
            elif equation_type == 'tan':
                general.append(f"{round(sol, 2)}° + 180°n")
        
        return jsonify({
            'principal_solutions': [round(s, 2) for s in solutions],
            'general_solutions': general
        })
    return render_template('solve_trig_equations.html')

# 11. Inverse Trigonometric Functions
@trig_bp.route('/inverse_trig', methods=['GET', 'POST'])
def inverse_trig():
    if request.method == 'POST':
        data = request.json
        function = data['function']
        value = float(data['value'])
        
        if function == 'asin':
            if -1 <= value <= 1:
                angle = math.degrees(math.asin(value))
            else:
                angle = "undefined"
        elif function == 'acos':
            if -1 <= value <= 1:
                angle = math.degrees(math.acos(value))
            else:
                angle = "undefined"
        elif function == 'atan':
            angle = math.degrees(math.atan(value))
        
        return jsonify({
            'function': function,
            'value': value,
            'angle': round(angle, 4) if isinstance(angle, float) else angle,
            'range': get_inverse_range(function)
        })
    return render_template('inverse_trig.html')

def get_inverse_range(function):
    ranges = {
        'asin': "-90° to 90°",
        'acos': "0° to 180°",
        'atan': "-90° to 90°"
    }
    return ranges.get(function, "")

# 12. Law of Sines
@trig_bp.route('/law_of_sines', methods=['GET', 'POST'])
def law_of_sines():
    if request.method == 'POST':
        data = request.json
        side_a = float(data.get('a', 0)) if data.get('a') not in [None, ''] else 0
        side_b = float(data.get('b', 0)) if data.get('b') not in [None, ''] else 0
        side_c = float(data.get('c', 0)) if data.get('c') not in [None, ''] else 0
        angle_a = float(data.get('A', 0)) if data.get('A') not in [None, ''] else 0
        angle_b = float(data.get('B', 0)) if data.get('B') not in [None, ''] else 0
        angle_c = float(data.get('C', 0)) if data.get('C') not in [None, ''] else 0
        
        # Calculate missing values
        if side_a and angle_a:
            if side_b and not angle_b:
                angle_b = math.degrees(math.asin((side_b * math.sin(math.radians(angle_a))) / side_a))
            elif angle_b and not side_b:
                side_b = (side_a * math.sin(math.radians(angle_b))) / math.sin(math.radians(angle_a))
            
            if side_c and not angle_c:
                angle_c = math.degrees(math.asin((side_c * math.sin(math.radians(angle_a))) / side_a))
            elif angle_c and not side_c:
                side_c = (side_a * math.sin(math.radians(angle_c))) / math.sin(math.radians(angle_a))
            
            # Calculate third angle if two are known
            if angle_a and angle_b and not angle_c:
                angle_c = 180 - angle_a - angle_b
            elif angle_a and angle_c and not angle_b:
                angle_b = 180 - angle_a - angle_c
        
        # Validate triangle angles
        if not validate_triangle_angles(angle_a, angle_b, angle_c):
            return jsonify({'error': 'Angles do not sum to 180°'})
        
        return jsonify({
            'a': round(side_a, 4) if side_a else None,
            'b': round(side_b, 4) if side_b else None,
            'c': round(side_c, 4) if side_c else None,
            'A': round(angle_a, 4) if angle_a else None,
            'B': round(angle_b, 4) if angle_b else None,
            'C': round(angle_c, 4) if angle_c else None
        })
    return render_template('law_of_sines.html')

# 13. Law of Cosines
@trig_bp.route('/law_of_cosines', methods=['GET', 'POST'])
def law_of_cosines():
    if request.method == 'POST':
        data = request.json
        side_a = float(data.get('a', 0)) if data.get('a') not in [None, ''] else 0
        side_b = float(data.get('b', 0)) if data.get('b') not in [None, ''] else 0
        side_c = float(data.get('c', 0)) if data.get('c') not in [None, ''] else 0
        angle_c = float(data.get('C', 0)) if data.get('C') not in [None, ''] else 0
        
        # Calculate missing values
        if side_a and side_b and side_c:
            # Find angle C
            angle_c = math.degrees(math.acos((side_a**2 + side_b**2 - side_c**2) / (2 * side_a * side_b)))
        elif side_a and side_b and angle_c:
            # Find side c
            side_c = math.sqrt(side_a**2 + side_b**2 - 2 * side_a * side_b * math.cos(math.radians(angle_c)))
        
        return jsonify({
            'a': round(side_a, 4) if side_a else None,
            'b': round(side_b, 4) if side_b else None,
            'c': round(side_c, 4) if side_c else None,
            'C': round(angle_c, 4) if angle_c else None
        })
    return render_template('law_of_cosines.html')

# 14. Area of Triangle using Trigonometry
@trig_bp.route('/area_triangle', methods=['GET', 'POST'])
def area_triangle():
    if request.method == 'POST':
        data = request.json
        side_a = float(data.get('a', 0)) if data.get('a') not in [None, ''] else 0
        side_b = float(data.get('b', 0)) if data.get('b') not in [None, ''] else 0
        angle_c = float(data.get('C', 0)) if data.get('C') not in [None, ''] else 0
        
        if side_a and side_b and angle_c:
            area = 0.5 * side_a * side_b * math.sin(math.radians(angle_c))
        else:
            area = None
        
        return jsonify({
            'a': round(side_a, 4) if side_a else None,
            'b': round(side_b, 4) if side_b else None,
            'C': round(angle_c, 4) if angle_c else None,
            'area': round(area, 4) if area else None
        })
    return render_template('area_triangle.html')

# 15. Trigonometric Form of Complex Numbers
@trig_bp.route('/complex-numbers', methods=['GET', 'POST'])
def complex_numbers():
    if request.method == 'POST':
        data = request.json
        real = float(data['real'])
        imag = float(data['imag'])
        
        z = complex(real, imag)
        r = abs(z)
        theta = math.degrees(cmath.phase(z))
        
        return jsonify({
            'rectangular': f"{real} + {imag}i",
            'polar': f"{round(r, 4)} (cos {round(theta, 4)}° + i sin {round(theta, 4)}°)",
            'magnitude': round(r, 4),
            'angle': round(theta, 4)
        })
    return render_template('trig/complex_numbers.html')