from flask import Blueprint, render_template, request

mensuration = Blueprint('mensuration', __name__, template_folder='templates/men')

# Define available shapes and their formulas
shapes = {
    "circle": {"area": "Area = π × r²", "perimeter": "Perimeter = 2πr"},
    "square": {"area": "Area = a²", "perimeter": "Perimeter = 4a"},
    "triangle": {"area": "Area = ½ × base × height"},
    "rectangle": {"area": "Area = length × breadth", "perimeter": "Perimeter = 2(length + breadth)"},
    "trapezium": {"area": "Area = ½ × (a + b) × height", "perimeter": "Perimeter = a + b + c + d"},
    "parallelogram": {"area": "Area = base × height"},
    "sector": {"area": "Area = (θ/360) × π × r²", "perimeter": "Perimeter = 2r + (θ/360)×2πr"},
    "cone": {"area": "Surface Area = πr(l+r)", "volume": "Volume = (1/3)πr²h"},
    "cylinder": {"area": "Surface Area = 2πr(h+r)", "volume": "Volume = πr²h"},
    "cube": {"area": "Surface Area = 6a²", "volume": "Volume = a³", "perimeter": "Perimeter = 12a"},
    "cuboid": {"area": "Surface Area = 2(lb + bh + hl)", "volume": "Volume = l × b × h"},
    "prism": {"volume": "Volume = Base Area × height"},
    "sphere": {"area": "Surface Area = 4πr²", "volume": "Volume = (4/3)πr³"},
    "rhombus": {"area": "Area = (p × q) ÷ 2", "perimeter": "Perimeter = 4a"},
    "pentagon": {"area": "Area = (5/4) × a² × cot(π/5)", "perimeter": "Perimeter = 5a"},
    "hexagon": {"area": "Area = (3√3 /2) × a²", "perimeter": "Perimeter = 6a"},
    "arc": {"length": "Length = (θ/360) × 2πr"},
    "ellipse": {"area": "Area = πab", "perimeter": "Perimeter ≈ 2π√((a²+b²)/2)"},
    "rectangular_prism": {"area": "Surface Area = 2(lw + lh + wh)", "volume": "Volume = l × w × h"},
    "square_pyramid": {"area": "Surface Area = a² + 2a√(a²/4 + h²)", "volume": "Volume = (1/3)a²h"},
    "square_pyramid_frustum": {"volume": "Volume = (1/3)h(a1² + a2² + (a1×a2))"},
    "conical_frustum": {
        "lateral_area": "Lateral Area = π(R + r)l",
        "area": "Total Area = π(R² + r² + (R+r)l)",
        "volume": "Volume = (1/3)πh(R² + Rr + r²)"
    },
    "spherical_cap": {
        "volume": "Volume = (1/3)πh²(3R - h)",
        "area": "Curved Surface Area = 2πRh"
    },
    "spherical_segment": {
        "volume": "Volume = (1/6)πh(3a² + 3b² + h²)",
        "area": "Surface Area = 2πRh + πh²"
    },
    "ellipsoid": {"volume": "Volume = (4/3)πabc", "area": "Approximate Area = 4π((a^p b^p + a^p c^p + b^p c^p)/3)^(1/p)"},
}

# Home route showing shape options
@mensuration.route('/')
def home():
    return render_template('men/home.html', shapes=shapes)

# General calculator page (lists all shapes)
@mensuration.route('/calculator')
def calculator_home():
    return render_template('men/home.html', shapes=shapes)

# Core calculator with GET and POST
@mensuration.route('/calculator/<shape>/<calculation>', methods=['GET', 'POST'])
def calculator(shape, calculation):
    formula = shapes.get(shape, {}).get(calculation, "Formula not found")
    result = None

    if request.method == 'POST':
        try:
            if shape == "circle":
                r = float(request.form['r'])
                result = 3.1416 * r * r if calculation == "area" else 2 * 3.1416 * r

            elif shape == "square":
                a = float(request.form['a'])
                result = a * a if calculation == "area" else 4 * a

            elif shape == "triangle" and calculation == "area":
                base = float(request.form['base'])
                height = float(request.form['height'])
                result = 0.5 * base * height

            elif shape == "rectangle":
                length = float(request.form['length'])
                breadth = float(request.form['breadth'])
                result = length * breadth if calculation == "area" else 2 * (length + breadth)

            elif shape == "trapezium":
                if calculation == "area":
                    a = float(request.form['a'])
                    b = float(request.form['b'])
                    height = float(request.form['height'])
                    result = 0.5 * (a + b) * height
                else:
                    a = float(request.form['a'])
                    b = float(request.form['b'])
                    c = float(request.form['c'])
                    d = float(request.form['d'])
                    result = a + b + c + d

            elif shape == "parallelogram" and calculation == "area":
                base = float(request.form['base'])
                height = float(request.form['height'])
                result = base * height

            elif shape == "sector":
                r = float(request.form['r'])
                theta = float(request.form['theta'])
                if calculation == "area":
                    result = (theta / 360) * 3.1416 * r * r
                else:
                    result = ((theta / 360) * 2 * 3.1416 * r) + 2 * r

            elif shape == "cone":
                r = float(request.form['r'])
                h = float(request.form['h'])
                l = (r**2 + h**2) ** 0.5
                result = 3.1416 * r * (l + r) if calculation == "area" else (1/3) * 3.1416 * r * r * h

            elif shape == "cylinder":
                r = float(request.form['r'])
                h = float(request.form['h'])
                result = 2 * 3.1416 * r * (h + r) if calculation == "area" else 3.1416 * r * r * h

            elif shape == "cube":
                a = float(request.form['a'])
                if calculation == "area":
                    result = 6 * a * a
                elif calculation == "volume":
                    result = a**3
                elif calculation == "perimeter":
                    result = 12 * a

            elif shape == "cuboid":
                l = float(request.form['l'])
                b = float(request.form['b'])
                h = float(request.form['h'])
                result = 2 * (l*b + b*h + h*l) if calculation == "area" else l * b * h

            elif shape == "prism" and calculation == "volume":
                base_area = float(request.form['base_area'])
                height = float(request.form['height'])
                result = base_area * height

            elif shape == "sphere":
                r = float(request.form['r'])
                result = 4 * 3.1416 * r * r if calculation == "area" else (4/3) * 3.1416 * r**3

            elif shape == "rhombus":
                if calculation == "area":
                    p = float(request.form['p'])
                    q = float(request.form['q'])
                    result = (p * q) / 2
                else:
                    a = float(request.form['a'])
                    result = 4 * a

            elif shape == "pentagon":
                a = float(request.form['a'])
                import math
                result = (5/4) * a**2 * (1 / math.tan(math.pi/5)) if calculation == "area" else 5 * a

            elif shape == "hexagon":
                a = float(request.form['a'])
                import math
                result = (3 * (3)**0.5 / 2) * a * a if calculation == "area" else 6 * a

            elif shape == "arc" and calculation == "length":
                r = float(request.form['r'])
                theta = float(request.form['theta'])
                result = (theta / 360) * 2 * 3.1416 * r

            elif shape == "ellipse":
                a = float(request.form['a'])
                b = float(request.form['b'])
                result = 3.1416 * a * b if calculation == "area" else 2 * 3.1416 * ((a**2 + b**2)/2)**0.5

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('men/calculator.html', shape=shape, calculation=calculation, formula=formula, result=result)
