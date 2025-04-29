from flask import Flask, render_template, request
from sympy import symbols, Eq, solve, simplify, expand, factor, parse_expr, Function, dsolve, Derivative
from sympy.stats import Normal, P, E, variance, sample
import numpy as np
import scipy.stats as stats
import math
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Home
@app.route('/', methods=['GET', 'POST'])
def statistics():
    return render_template('/statistics/base.html')

# 1. Descriptive Statistics
@app.route('/statistics/descriptive', methods=['GET', 'POST'])
def descriptive_stats():
    result = None
    error = None
    if request.method == 'POST':
        data_str = request.form.get('data', '').strip()
        if not data_str:
            error = "Please enter data values"
        else:
            try:
                # Parse comma-separated data
                data = [float(x.strip()) for x in data_str.split(',') if x.strip()]
                
                # Calculate statistics
                n = len(data)
                mean = sum(data) / n
                sorted_data = sorted(data)
                
                # Median
                if n % 2 == 1:
                    median = sorted_data[n//2]
                else:
                    median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                
                # Mode
                freq = {}
                for x in data:
                    freq[x] = freq.get(x, 0) + 1
                max_freq = max(freq.values())
                modes = [k for k, v in freq.items() if v == max_freq]
                
                # Variance and Standard Deviation
                variance = sum((x - mean)**2 for x in data) / (n - 1)  # Sample variance
                std_dev = math.sqrt(variance)
                
                # Quartiles
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                
                # Min, Max, Range
                min_val = min(data)
                max_val = max(data)
                data_range = max_val - min_val
                
                result = {
                    'n': n,
                    'mean': round(mean, 4),
                    'median': round(median, 4),
                    'mode': modes,
                    'variance': round(variance, 4),
                    'std_dev': round(std_dev, 4),
                    'q1': round(q1, 4),
                    'q3': round(q3, 4),
                    'iqr': round(iqr, 4),
                    'min': min_val,
                    'max': max_val,
                    'range': data_range
                }
                
            except ValueError:
                error = "Invalid data format. Use comma-separated numbers (e.g., 1, 2, 3.5)"
            except Exception as e:
                error = f"Error calculating statistics: {str(e)}"
    
    return render_template('statistics/descriptive.html', result=result, error=error)

# 2. Probability Distributions
@app.route('/statistics/distributions', methods=['GET', 'POST'])
def probability_distributions():
    result = None
    error = None
    if request.method == 'POST':
        dist_type = request.form.get('distribution', 'normal')
        try:
            if dist_type == 'normal':
                mean = float(request.form.get('mean', 0))
                std_dev = float(request.form.get('std_dev', 1))
                x = float(request.form.get('x_value', 0))
                
                # Calculate PDF and CDF
                pdf = stats.norm.pdf(x, mean, std_dev)
                cdf = stats.norm.cdf(x, mean, std_dev)
                
                result = {
                    'type': 'Normal',
                    'parameters': f"μ={mean}, σ={std_dev}",
                    'x': x,
                    'pdf': round(pdf, 6),
                    'cdf': round(cdf, 6)
                }
                
            elif dist_type == 'binomial':
                n = int(request.form.get('trials', 10))
                p = float(request.form.get('prob', 0.5))
                k = int(request.form.get('successes', 5))
                
                # Calculate probability
                pmf = stats.binom.pmf(k, n, p)
                cdf = stats.binom.cdf(k, n, p)
                
                result = {
                    'type': 'Binomial',
                    'parameters': f"n={n}, p={p}",
                    'k': k,
                    'pmf': round(pmf, 6),
                    'cdf': round(cdf, 6)
                }
                
            elif dist_type == 'poisson':
                lmbda = float(request.form.get('lambda', 3))
                k = int(request.form.get('events', 2))
                
                # Calculate probability
                pmf = stats.poisson.pmf(k, lmbda)
                cdf = stats.poisson.cdf(k, lmbda)
                
                result = {
                    'type': 'Poisson',
                    'parameters': f"λ={lmbda}",
                    'k': k,
                    'pmf': round(pmf, 6),
                    'cdf': round(cdf, 6)
                }
                
        except ValueError:
            error = "Invalid parameter values"
        except Exception as e:
            error = f"Error calculating distribution: {str(e)}"
    
    return render_template('statistics/distributions.html', result=result, error=error)

# 3. Hypothesis Testing
@app.route('/statistics/hypothesis', methods=['GET', 'POST'])
def hypothesis_testing():
    result = None
    error = None
    if request.method == 'POST':
        test_type = request.form.get('test_type', 'z_test')
        try:
            if test_type == 'z_test':
                # One-sample z-test
                sample_mean = float(request.form.get('sample_mean'))
                pop_mean = float(request.form.get('pop_mean'))
                pop_std = float(request.form.get('pop_std'))
                n = int(request.form.get('sample_size'))
                alpha = float(request.form.get('alpha', 0.05))
                
                # Calculate z-score and p-value
                z_score = (sample_mean - pop_mean) / (pop_std / math.sqrt(n))
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                # Determine significance
                significant = p_value < alpha
                
                result = {
                    'test': 'One-Sample Z-Test',
                    'z_score': round(z_score, 4),
                    'p_value': round(p_value, 6),
                    'alpha': alpha,
                    'significant': significant,
                    'conclusion': "Reject H₀" if significant else "Fail to reject H₀"
                }
                
            elif test_type == 't_test':
                # One-sample t-test
                sample_data = [float(x.strip()) for x in request.form.get('sample_data', '').split(',') if x.strip()]
                pop_mean = float(request.form.get('pop_mean'))
                alpha = float(request.form.get('alpha', 0.05))
                
                n = len(sample_data)
                sample_mean = sum(sample_data) / n
                sample_std = math.sqrt(sum((x - sample_mean)**2 for x in sample_data) / (n - 1))
                
                # Calculate t-score and p-value
                t_score = (sample_mean - pop_mean) / (sample_std / math.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_score), df=n-1))
                
                # Determine significance
                significant = p_value < alpha
                
                result = {
                    'test': 'One-Sample T-Test',
                    't_score': round(t_score, 4),
                    'p_value': round(p_value, 6),
                    'df': n-1,
                    'alpha': alpha,
                    'significant': significant,
                    'conclusion': "Reject H₀" if significant else "Fail to reject H₀"
                }
                
        except ValueError:
            error = "Invalid input values"
        except Exception as e:
            error = f"Error performing test: {str(e)}"
    
    return render_template('statistics/hypothesis.html', result=result, error=error)

# 4. Confidence Intervals
@app.route('/statistics/confidence', methods=['GET', 'POST'])
def confidence_intervals():
    result = None
    error = None
    if request.method == 'POST':
        ci_type = request.form.get('ci_type', 'mean')
        try:
            if ci_type == 'mean':
                # Mean CI
                sample_mean = float(request.form.get('sample_mean'))
                sample_std = float(request.form.get('sample_std'))
                n = int(request.form.get('sample_size'))
                confidence = float(request.form.get('confidence', 95)) / 100
                
                # Calculate margin of error
                if n >= 30 or request.form.get('pop_std'):
                    # Z-interval
                    pop_std = float(request.form.get('pop_std', sample_std))
                    z_critical = stats.norm.ppf((1 + confidence) / 2)
                    margin = z_critical * (pop_std / math.sqrt(n))
                    method = "Z-interval (σ known or n ≥ 30)"
                else:
                    # T-interval
                    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
                    margin = t_critical * (sample_std / math.sqrt(n))
                    method = "T-interval (σ unknown and n < 30)"
                
                lower = sample_mean - margin
                upper = sample_mean + margin
                
                result = {
                    'type': 'Mean',
                    'method': method,
                    'estimate': sample_mean,
                    'lower': round(lower, 4),
                    'upper': round(upper, 4),
                    'margin': round(margin, 4),
                    'confidence': f"{confidence*100}%"
                }
                
            elif ci_type == 'proportion':
                # Proportion CI
                p_hat = float(request.form.get('sample_proportion'))
                n = int(request.form.get('sample_size'))
                confidence = float(request.form.get('confidence', 95)) / 100
                
                # Check conditions
                if n * p_hat < 10 or n * (1 - p_hat) < 10:
                    error = "Conditions not met: np̂ and n(1-p̂) should both be ≥ 10"
                else:
                    z_critical = stats.norm.ppf((1 + confidence) / 2)
                    margin = z_critical * math.sqrt(p_hat * (1 - p_hat) / n)
                    
                    lower = p_hat - margin
                    upper = p_hat + margin
                    
                    result = {
                        'type': 'Proportion',
                        'estimate': p_hat,
                        'lower': round(lower, 4),
                        'upper': round(upper, 4),
                        'margin': round(margin, 4),
                        'confidence': f"{confidence*100}%"
                    }
                    
        except ValueError:
            error = "Invalid input values"
        except Exception as e:
            error = f"Error calculating interval: {str(e)}"
    
    return render_template('statistics/confidence.html', result=result, error=error)

# 5. Linear Regression
@app.route('/statistics/regression', methods=['GET', 'POST'])
def linear_regression():
    result = None
    error = None
    if request.method == 'POST':
        try:
            x_str = request.form.get('x_data', '').strip()
            y_str = request.form.get('y_data', '').strip()
            
            x_data = [float(x.strip()) for x in x_str.split(',') if x.strip()]
            y_data = [float(y.strip()) for y in y_str.split(',') if y.strip()]
            
            if len(x_data) != len(y_data):
                error = "X and Y data must have same number of values"
            elif len(x_data) < 2:
                error = "At least 2 data points required"
            else:
                # Calculate regression
                n = len(x_data)
                sum_x = sum(x_data)
                sum_y = sum(y_data)
                sum_xy = sum(x * y for x, y in zip(x_data, y_data))
                sum_x2 = sum(x**2 for x in x_data)
                sum_y2 = sum(y**2 for y in y_data)
                
                # Slope (b1) and intercept (b0)
                numerator = (n * sum_xy) - (sum_x * sum_y)
                denominator = (n * sum_x2) - (sum_x ** 2)
                slope = numerator / denominator
                intercept = (sum_y - slope * sum_x) / n
                
                # Calculate R-squared
                ss_total = sum((y - (sum_y/n))**2 for y in y_data)
                ss_res = sum((y - (intercept + slope*x))**2 for x, y in zip(x_data, y_data))
                r_squared = 1 - (ss_res / ss_total)
                
                # Standard error
                std_error = math.sqrt(ss_res / (n - 2))
                
                result = {
                    'equation': f"y = {round(intercept, 4)} + {round(slope, 4)}x",
                    'slope': round(slope, 4),
                    'intercept': round(intercept, 4),
                    'r_squared': round(r_squared, 4),
                    'std_error': round(std_error, 4),
                    'n': n
                }
                
        except ValueError:
            error = "Invalid data format. Use comma-separated numbers"
        except ZeroDivisionError:
            error = "Cannot compute regression (division by zero)"
        except Exception as e:
            error = f"Error performing regression: {str(e)}"
    
    return render_template('statistics/regression.html', result=result, error=error)

# 6. Chi-Square Test
@app.route('/statistics/chi_square', methods=['GET', 'POST'])
def chi_square_test():
    result = None
    error = None
    if request.method == 'POST':
        test_type = request.form.get('chi_test_type', 'goodness')
        try:
            if test_type == 'goodness':
                # Goodness of fit test
                observed = [float(x.strip()) for x in request.form.get('observed', '').split(',') if x.strip()]
                expected = [float(x.strip()) for x in request.form.get('expected', '').split(',') if x.strip()]
                
                if len(observed) != len(expected):
                    error = "Observed and expected must have same number of categories"
                elif any(e == 0 for e in expected):
                    error = "Expected values cannot be zero"
                else:
                    chi2, p = stats.chisquare(observed, f_exp=expected)
                    df = len(observed) - 1
                    
                    result = {
                        'test': 'Chi-Square Goodness of Fit',
                        'chi2': round(chi2, 4),
                        'p_value': round(p, 6),
                        'df': df,
                        'conclusion': "Reject H₀ (distributions differ)" if p < 0.05 else "Fail to reject H₀ (distributions similar)"
                    }
                    
            elif test_type == 'independence':
                # Test for independence
                # This would require a more complex input (contingency table)
                # Simplified for example purposes
                error = "Contingency table input not implemented in this example"
                
        except ValueError:
            error = "Invalid input values"
        except Exception as e:
            error = f"Error performing test: {str(e)}"
    
    return render_template('statistics/chi_square.html', result=result, error=error)

# 7. ANOVA
@app.route('/statistics/anova', methods=['GET', 'POST'])
def anova_test():
    result = None
    error = None
    if request.method == 'POST':
        try:
            # Get up to 3 groups of data
            groups = []
            for i in range(1, 4):
                data_str = request.form.get(f'group{i}', '').strip()
                if data_str:
                    group_data = [float(x.strip()) for x in data_str.split(',') if x.strip()]
                    if group_data:
                        groups.append(group_data)
            
            if len(groups) < 2:
                error = "At least 2 groups required"
            else:
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                result = {
                    'test': 'One-Way ANOVA',
                    'f_stat': round(f_stat, 4),
                    'p_value': round(p_value, 6),
                    'num_groups': len(groups),
                    'conclusion': "Reject H₀ (means differ)" if p_value < 0.05 else "Fail to reject H₀ (means similar)"
                }
                
        except ValueError:
            error = "Invalid data format. Use comma-separated numbers"
        except Exception as e:
            error = f"Error performing ANOVA: {str(e)}"
    
    return render_template('statistics/anova.html', result=result, error=error)

# 8. Correlation
@app.route('/statistics/correlation', methods=['GET', 'POST'])
def correlation():
    result = None
    error = None
    if request.method == 'POST':
        try:
            x_str = request.form.get('x_data', '').strip()
            y_str = request.form.get('y_data', '').strip()
            
            x_data = [float(x.strip()) for x in x_str.split(',') if x.strip()]
            y_data = [float(y.strip()) for y in y_str.split(',') if y.strip()]
            
            if len(x_data) != len(y_data):
                error = "X and Y data must have same number of values"
            elif len(x_data) < 2:
                error = "At least 2 data points required"
            else:
                # Calculate Pearson's r
                r, p_value = stats.pearsonr(x_data, y_data)
                
                result = {
                    'type': "Pearson's r",
                    'correlation': round(r, 4),
                    'p_value': round(p_value, 6),
                    'interpretation': interpret_correlation(r),
                    'n': len(x_data)
                }
                
        except ValueError:
            error = "Invalid data format. Use comma-separated numbers"
        except Exception as e:
            error = f"Error calculating correlation: {str(e)}"
    
    return render_template('statistics/correlation.html', result=result, error=error)

def interpret_correlation(r):
    """Helper function to interpret correlation coefficient"""
    abs_r = abs(r)
    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak or none"
    
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"
    return f"{direction} {strength} correlation"

# 9. Probability Calculator
@app.route('/statistics/probability', methods=['GET', 'POST'])
def probability_calculator():
    result = None
    error = None
    if request.method == 'POST':
        try:
            event_a = float(request.form.get('event_a', 0))
            event_b = float(request.form.get('event_b', 0))
            operation = request.form.get('operation', 'independent')
            
            if not (0 <= event_a <= 1) or not (0 <= event_b <= 1):
                error = "Probabilities must be between 0 and 1"
            else:
                if operation == 'independent':
                    # P(A and B) = P(A) * P(B)
                    result_prob = event_a * event_b
                    explanation = f"P(A and B) = P(A) × P(B) = {event_a} × {event_b}"
                elif operation == 'union':
                    # P(A or B) = P(A) + P(B) - P(A)P(B)
                    result_prob = event_a + event_b - (event_a * event_b)
                    explanation = f"P(A or B) = P(A) + P(B) - P(A)P(B) = {event_a} + {event_b} - ({event_a}×{event_b})"
                elif operation == 'conditional':
                    # P(A|B) = P(A and B) / P(B)
                    joint = float(request.form.get('joint_prob', 0))
                    if joint > event_b:
                        error = "P(A and B) cannot be greater than P(B)"
                    else:
                        result_prob = joint / event_b
                        explanation = f"P(A|B) = P(A and B) / P(B) = {joint} / {event_b}"
                
                if not error:
                    result = {
                        'operation': operation.replace('_', ' ').title(),
                        'probability': round(result_prob, 6),
                        'explanation': explanation
                    }
                    
        except ValueError:
            error = "Invalid probability values"
        except ZeroDivisionError:
            error = "Cannot divide by zero in conditional probability"
        except Exception as e:
            error = f"Error calculating probability: {str(e)}"
    
    return render_template('statistics/probability.html', result=result, error=error)

# 10. Sampling Distributions
@app.route('/statistics/sampling', methods=['GET', 'POST'])
def sampling_distributions():
    result = None
    error = None
    if request.method == 'POST':
        try:
            pop_mean = float(request.form.get('pop_mean', 0))
            pop_std = float(request.form.get('pop_std', 1))
            sample_size = int(request.form.get('sample_size', 30))
            sample_mean = float(request.form.get('sample_mean', pop_mean))
            
            # Calculate standard error
            std_error = pop_std / math.sqrt(sample_size)
            
            # Calculate z-score for sample mean
            z_score = (sample_mean - pop_mean) / std_error
            
            # Calculate probability of observing this sample mean or more extreme
            if z_score >= 0:
                p_value = 2 * (1 - stats.norm.cdf(z_score))
            else:
                p_value = 2 * stats.norm.cdf(z_score)
            
            result = {
                'pop_mean': pop_mean,
                'pop_std': pop_std,
                'sample_size': sample_size,
                'sample_mean': sample_mean,
                'std_error': round(std_error, 4),
                'z_score': round(z_score, 4),
                'p_value': round(p_value, 6)
            }
            
        except ValueError:
            error = "Invalid parameter values"
        except ZeroDivisionError:
            error = "Population standard deviation cannot be zero"
        except Exception as e:
            error = f"Error calculating sampling distribution: {str(e)}"
    
    return render_template('statistics/sampling.html', result=result, error=error)

# 11. Runge-Kutta Method (4th Order)
@app.route('/statistics/runge_kutta', methods=['GET', 'POST'])
def runge_kutta():
    result = None
    error = None
    plot_url = None
    
    if request.method == 'POST':
        try:
            # Get form inputs
            equation = request.form.get('equation', '').strip()
            initial_x = float(request.form.get('initial_x', 0))
            initial_y = float(request.form.get('initial_y', 1))
            step_size = float(request.form.get('step_size', 0.1))
            steps = int(request.form.get('steps', 10))
            
            # Parse the differential equation (simple format like "y' = x + y")
            if not equation:
                raise ValueError("Please enter a differential equation")
            
            if "=" not in equation:
                raise ValueError("Equation must contain '=' sign")
            
            # Implement 4th Order Runge-Kutta method
            def f(x, y):
                try:
                    # Simple evaluation of the right-hand side
                    rhs = equation.split('=')[1].strip()
                    return eval(rhs, {'x': x, 'y': y, 'math': math, 'np': np})
                except Exception as e:
                    raise ValueError(f"Could not evaluate equation: {str(e)}")
            
            # Perform Runge-Kutta integration
            x_values = [initial_x]
            y_values = [initial_y]
            
            for _ in range(steps):
                x = x_values[-1]
                y = y_values[-1]
                
                k1 = step_size * f(x, y)
                k2 = step_size * f(x + step_size/2, y + k1/2)
                k3 = step_size * f(x + step_size/2, y + k2/2)
                k4 = step_size * f(x + step_size, y + k3)
                
                y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
                x_new = x + step_size
                
                x_values.append(x_new)
                y_values.append(y_new)
            
            # Generate plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, 'b-', linewidth=2, label='Numerical Solution')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Runge-Kutta Method Solution')
            plt.grid(True)
            plt.legend()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('ascii')
            plt.close()
            
            # Prepare results
            result = {
                'equation': equation,
                'initial_conditions': f"y({initial_x}) = {initial_y}",
                'step_size': step_size,
                'steps': steps,
                'solution': list(zip(x_values, y_values)),
                'final_value': (x_values[-1], y_values[-1])
            }
            
        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Error solving ODE: {str(e)}"
    
    return render_template('statistics/runge_kutta.html', 
                         result=result, 
                         error=error,
                         plot_url=plot_url,
                         equation_value=request.form.get('equation', ''),
                         initial_x_value=request.form.get('initial_x', '0'),
                         initial_y_value=request.form.get('initial_y', '1'),
                         step_size_value=request.form.get('step_size', '0.1'),
                         steps_value=request.form.get('steps', '10'))

if __name__ == '__main__':
    app.run(debug=True)