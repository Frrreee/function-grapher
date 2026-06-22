from flask import Flask, render_template, request, jsonify
import sympy as sp
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    expression_str = data.get('expression', '')
    
    if not expression_str:
        return jsonify({'error': 'No expression provided'}), 400
    
    try:
        # Define symbols
        x, y = sp.symbols('x y')
        
        # Replace common JS math syntax with python math syntax if needed
        expression_str = expression_str.replace('^', '**')
        
        # Check for '=' to split into LHS and RHS for implicit equations
        if '=' in expression_str:
            parts = expression_str.split('=')
            if len(parts) == 2:
                # Rearrange LHS = RHS to LHS - (RHS) = 0
                expression_str = f"({parts[0]}) - ({parts[1]})"
            else:
                return jsonify({'error': 'Invalid equation: multiple "=" signs'}), 400
        
        # Parse the expression safely
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        expr = parse_expr(expression_str, transformations=transformations)
        
        free_syms = expr.free_symbols
        # If 'y' is a free symbol, evaluate it implicitly
        is_implicit = y in free_syms
        
        if is_implicit:
            # Generate 2D grid for contour plotting
            x_vals = np.linspace(-15, 15, 200)
            y_vals = np.linspace(-15, 15, 200)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Lambdify with both (x, y) variables
            f = sp.lambdify((x, y), expr, 'numpy')
            z_vals = f(X, Y)
            
            if np.isscalar(z_vals):
                z_vals = np.full_like(X, z_vals)
            elif z_vals.shape != X.shape:
                z_vals = np.broadcast_to(z_vals, X.shape)
                
            # Clean complex/infinite values for JSON
            z_flat = z_vals.flatten()
            z_clean = []
            for val in z_flat:
                try:
                    if isinstance(val, complex) or not np.isfinite(val):
                        z_clean.append(None)
                    else:
                        z_clean.append(float(val))
                except Exception:
                    z_clean.append(None)
            
            # Reshape back to 2D list of shape (len(y_vals), len(x_vals))
            z_clean_2d = [z_clean[i:i + len(x_vals)] for i in range(0, len(z_clean), len(x_vals))]
            
            return jsonify({
                'type': 'implicit',
                'x': x_vals.tolist(),
                'y': y_vals.tolist(),
                'z': z_clean_2d
            })
        else:
            # Explicit: f(x)
            f = sp.lambdify(x, expr, 'numpy')
            x_vals = np.linspace(-20, 20, 800)
            y_vals = f(x_vals)
            
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)
                
            y_vals_clean = []
            for y_val in y_vals:
                try:
                    if isinstance(y_val, complex) or not np.isfinite(y_val):
                        y_vals_clean.append(None)
                    else:
                        y_vals_clean.append(float(y_val))
                except Exception:
                    y_vals_clean.append(None)
                    
            return jsonify({
                'type': 'explicit',
                'x': x_vals.tolist(),
                'y': y_vals_clean
            })
        
    except Exception as e:
        return jsonify({'error': f'Invalid mathematical expression: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
