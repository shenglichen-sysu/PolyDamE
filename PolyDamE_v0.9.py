# PolyDamE: A deterministic open-source code for
# damage energy calculations in arbitrary polyatomic materials
#
# PolyDamE is writen in and runs with Python3
#
# A new version with more detailed comments will be uploaded soon
#
# MIT License
#
# Copyright (c) 2026 Shengli Chen
#
# Sun Yat-sen University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

import numpy as np
from scipy.interpolate import CubicHermiteSpline
# from scipy.integrate import simps as simpson
from scipy.integrate import quad
from scipy.optimize import root
import matplotlib.pyplot as plt

# ==========================================
# 1. Constants and Helper Functions
# ==========================================

# Physical Constants from Paper/Docx
LAMBDA_WSS = 1.309 # Winterbon lambda
CONST_34 = 34.8552
CONST_U = 2.646e-4

def get_atomic_data(name):
    """
    Returns (Z, A) for given element name.
    Add more elements as needed.
    Can be read data from any tables
    """
    data = {
        'O': (8, 15.999),
        'Al': (13, 26.982),
        'Si': (14, 28.086),
        'Ti': (22, 47.867),
        'Nb': (41, 92.906),
        'Y': (39, 88.906),
        'U': (92, 238.029),
        'C': (6, 12.011),
        'Zr': (40, 91.224),
        'Mg': (12, 24.305),
        'Fe': (26, 55.845),
        'Cr': (24, 51.996),
        'Ni': (28, 58.693),
        'P': (15, 30.974),
        'V': (23, 50.942),
        'Mo': (42, 95.94),
        'Mn': (25, 54.938),
        'S': (16, 32.065)
    }
    return data.get(name)

def calc_parameters(elements, stoichiometry):
    """
    Calculate G_ij, M_ij, U_ij matrices.
    elements: list of tuples [(Z, A), ...]
    stoichiometry: list of relative amounts [n1, n2, ...]
    stoichiometry is automatically normalized
    """
    n = len(elements)
    Z = np.array([e[0] for e in elements])
    A = np.array([e[1] for e in elements])
    N_frac = np.array(stoichiometry) / sum(stoichiometry)
    
    G = np.zeros((n, n))
    U = np.zeros((n, n))
    M = np.zeros((n, n))
    
    # Comments need to be updated ...
    # 
    # Calculate G_ij (PDF Eq 3 / Docx Eq 3)
    # 
    # Denominator term sum_k (N_k/N * Z_k / (Z_i^(2/3) + Z_k^(2/3))^(3/2))
    #
    for i in range(n):
        denom_sum = 0.0
        for k in range(n):
            term = (N_frac[k] * Z[k]) / ((Z[i]**(2/3) + Z[k]**(2/3))**1.5)
            denom_sum += term
            
        for j in range(n):
            # Numerator
            num = 34.8552 * (A[i] / (Z[i]**(1/6))) * N_frac[j] * (Z[j] / (A[j]**0.5))
            num /= (Z[i]**(2/3) + Z[j]**(2/3))**0.5
            G[i, j] = num / denom_sum

    # Calculate U_ij (PDF Eq 4 / Docx Eq 4)
    for i in range(n):
        for j in range(n):
            term = 2.646e-4 * (A[j] / A[i]) * (1.0 / (Z[i]**2 * Z[j]**2 * (Z[i]**(2/3) + Z[j]**(2/3))))
            U[i, j] = term

    # Calculate M_ij (Mass transfer factor)
    for i in range(n):
        for j in range(n):
            M[i, j] = (4 * A[i] * A[j]) / ((A[i] + A[j])**2)
            
    return G, U, M

def f_winterbon(xi):
    """
    The screening function f(xi) using Winterbon approximation.
    Note: f(t^1/2). Let xi = t^1/2 = (U*E*T)^1/2 in Coulter's paper.
    Then t = xi^2.
    Formula: lambda * t^(1/6) * [1 + (2*lambda*t^(2/3))^(2/3)]^(-3/2)
           = lambda * xi^(1/3) * [1 + (2*lambda*xi^(4/3))^(2/3)]^(-3/2)
    """
    lam = LAMBDA_WSS
    
    term1 = lam * xi**(1.0/3.0)
    term2 = (1.0 + (2 * lam * xi**(4.0/3.0))**(2.0/3.0))**(-1.5)
    return term1 * term2

def f_coulter(xi):
    """
    Coulter & Parkin f_CP(xi) approximation (eq. (8) in their paper).
    Only used for comparison
    No need for normal applications
    """
    if xi <= 0.0:
        return 0.0
    u = xi ** 0.3375
    # coefficients from Coulter eq.(8)
    a0 = 1.43
    # polynomial: (1 - 0.30925*u + 2.6989*u^2 - 1.03887*u^3 + 2.86*u^4)
    p1 = -0.30925
    p2 =  2.6989
    p3 = -1.03887
    p4 =  2.86
    poly = 1.0 + p1 * u + p2 * (u**2) + p3 * (u**3) + p4 * (u**4)
    return a0 * u**4/xi / poly

def integrand_kernel(T, E_curr, nu_func_j, nu_func_i, i, j, U_val):
    """
    Computes the term inside the integral: 
    (f(xi) / 2T^1.5) * [nu_j(T) + nu_i(E-T) - nu_i(E)]
    """
    # xi = sqrt(U * E * T)
    xi = np.sqrt(U_val * E_curr * T)
    f_val = f_winterbon(xi)
    
    # Bracket term
    # nu_func_i/j are callables (interpolators)
    val_j_T = nu_func_j(T)
    val_i_E_minus_T = nu_func_i(E_curr - T)
    val_i_E = nu_func_i(E_curr)
    
    bracket = val_j_T + val_i_E_minus_T - val_i_E
    
    # Kernel: f / (2 * T^1.5)
    # Note: T=0 singularity is handled by the bracket vanishing ~T
    # Code handles T close to 0 safely
    kernel = f_val / (2 * np.maximum(T, 1e-20)**1.5)
    
    return kernel * bracket

def integrand_kernel_for_quad(T, E_curr, nu_func_j, nu_func_i, nu_i_E_curr, U_val):
    """
    Computes the term inside the integral for scipy.integrate.quad.
    (f(xi) / 2T^1.5) * [nu_j(T) + nu_i(E_curr - T) - nu_i_E_curr]
    E_curr here is the current energy point E_next.
    nu_i_E_curr is nu_i(E_next), passed as a value to avoid re-evaluation.
    """
    # xi = sqrt(U * E * T)
    xi = np.sqrt(U_val * E_curr * T)
    f_val = f_winterbon(xi)
    
    # Bracket term
    # nu_func_i/j are callables (interpolators)
    val_j_T = nu_func_j(T)
    
    # Check for T > E_curr, which shouldn't happen but is a safety check for spline evaluation
    E_minus_T = E_curr - T
    if E_minus_T < 0:
        val_i_E_minus_T = 0.0 # nu(E) is defined as 0 for E<0
    else:
        val_i_E_minus_T = nu_func_i(E_minus_T)
        
    bracket = val_j_T + val_i_E_minus_T - nu_i_E_curr
    
    # Kernel: f / (2 * T^1.5)
    # T_safe ensures the denominator is non-zero (even though quad handles the singularity)
    T_safe = np.maximum(T, 1e-20) 
    kernel = f_val / (2 * T_safe**1.5)
    
    return kernel * bracket


# ==========================================
# 2. Initialization (Method 2 / Eq 22 in our original doc)
# to be updated ...
# E_min is E_0 in the doc
# ==========================================

def solve_initial_slopes(E_min, G, U, M, n_types):
    """
    Solves the linear system for nu'_i(E) at small E based on 
    Eq (22) from the docx (Method 2 for small E).
    
    Equation derived from docx:
    E^{1/6} * nu'_i = (lambda/2) * sum_j { G_ij * U_ij^(1/6) * [ 
        (3/10 M^5/3 - 3/2 M^2/3) * nu'_i + 
        (3/10 M^5/3) * nu'_j + 
        (3/2 M^2/3 - 3/5 M^5/3) 
    ] }
    
    Rearranged to A * nu' = B
    We know the conditions at E = 0
    Here, we estimate the values at sufficient small E_min > 0
    """
    A_mat = np.zeros((n_types, n_types))
    B_vec = np.zeros(n_types)
    lam = LAMBDA_WSS
    
    for i in range(n_types):
        A_mat[i, i] = E_min**(1./6) # LHS term
        
        for j in range(n_types):
            m_val = M[i, j]
            u_val = U[i, j]
            g_val = G[i, j]
            
            # Common prefactor
            # Note: U^(1/6) is factorized
            prefactor = (lam / 2.0) * g_val * (u_val**(1.0/6.0))
            
            # Coefficients from Eq 22
            c_nu_i = (3/10) * m_val**(5/3) - (3/2) * m_val**(2/3)
            c_nu_j = (3/10) * m_val**(5/3)
            c_const = (3/2) * m_val**(2/3) - (3/5) * m_val**(5/3)
            
            # Fill matrix A (terms involving nu')
            # LHS was nu'_i, so subtract RHS nu'_i terms
            A_mat[i, i] -= prefactor * c_nu_i
            A_mat[i, j] -= prefactor * c_nu_j
            
            # Fill vector B (constant terms)
            B_vec[i] += prefactor * c_const
            
    # Solve linear system A*nu' = B
    nu_primes = np.linalg.solve(A_mat, B_vec)
    return nu_primes

# ==========================================
# 3. Main Solver
# ==========================================

def solve_polyatomic_damage_energy(material_name, elements, stoichiometry, 
                                   E_min=1e-4, E_max=1e6, steps_per_decade=10):
    
    # Setup Data
    n_types = len(elements)
    G, U, M = calc_parameters(elements, stoichiometry)
    
    # Create Grid
    num_decades = np.log10(E_max) - np.log10(E_min)
    n_steps = int(num_decades * steps_per_decade) + 1
    E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_steps)
    
    # Storage for solution: nu[atom_type][energy_index]
    # We also store derivatives for Spline construction
    nu_vals = np.zeros((n_types, n_steps))
    nu_primes = np.zeros((n_types, n_steps))
    
    # --- Initialization at E_min (Method 2) ---
    # nu'(E_min) from Eq 22
    # nu(E_min) approx E_min * (nu'(0) + nu'(E_min))/2
    # nu'(0) = 1, nu(0) = 0
    
    init_slopes = solve_initial_slopes(E_min, G, U, M, n_types)
    
    for i in range(n_types):
        nu_primes[i, 0] = init_slopes[i]
        nu_vals[i, 0] = E_min * (nu_primes[i, 0]+1)/2 # Approximation at low E <= E_min
        
        
    print(f"Initialization at {E_min:.2e} eV:")
    for i in range(n_types):
        print(f"  Atom {i}: nu={nu_vals[i,0]:.4e}, nu'={nu_primes[i,0]:.4f}")

    # --- Integration Loop ---
    # We step from k to k+1
    
    for k in range(n_steps - 1):
        E_curr = E_grid[k]
        E_next = E_grid[k+1]
        h = E_next - E_curr
        
        # Build splines for history [0, E_curr]
        # We append a temporary point for interpolation safety, but constraints apply
        # For history integration, we only need values up to E_curr defined.
        # But implicit equation needs evaluations up to E_next inside integrals?
        # Actually, integrals are up to M*E_next. Since M<=1, M*E_next <= E_next.
        # So we need a spline valid up to E_next.
        
        def residual_func(x):
            """
            x contains [nu'_0, nu'_1, ... nu'_{n-1}] at E_next.
            We calculate nu_next using Hermite interpolation formula.
            Then calculate residuals of Eq (8).
            """
            current_primes = x
            current_vals = np.zeros(n_types)
            
            # 1. Reconstruct nu(E_next) from nu(E_curr), nu'(E_curr), and guess nu'(E_next)
            # Cubic Hermite: nu_next = nu_curr + h/2 * (nu'_curr + nu'_next) + ... 
            # Standard cubic hermite basis on [0, h]:
            # p(t) = h00*y0 + h10*m0 + h01*y1 + h11*m1
            # But simpler here: standard integration rule implies:
            # nu_next = nu_curr + (h/2)*(nu'_curr + nu'_next) (Trapezoidal-ish projection for value)
            # This may be improved using Cubic Hermite polynomial approximations. 
            # This implies the curve is fully defined.
            # Value at E_next is constrained by continuity.
            # Using Simpson/Trapezoidal relation for derivative:
            # nu_next = nu_vals[i, k] + (h/2.0) * (nu_primes[i, k] + current_primes[i])
            
            for i in range(n_types):
                current_vals[i] = nu_vals[i, k] + (h/2.0) * (nu_primes[i, k] + current_primes[i])
            
            # 2. Build temporary splines valid up to E_next
            temp_splines = []
            for i in range(n_types):
                # Concatenate current history with guess
                # Note: This is slow in a loop, but robust
                # e_hist = np.append(E_grid[:k+1], E_next)
                # v_hist = np.append(nu_vals[i, :k+1], current_vals[i])
                # p_hist = np.append(nu_primes[i, :k+1], current_primes[i])

                # Concatenate history: [0, E_min, E_1, ..., E_curr, E_next]
                e_hist = np.concatenate(([0.0], E_grid[:k+1], [E_next]))
                
                # Values: [0, nu(E_min), ..., nu(E_curr), nu(E_next)]
                v_hist = np.concatenate(([0.0], nu_vals[i, :k+1], [current_vals[i]]))
                
                # Primes: [nu'(0), nu'(E_min), ..., nu'(E_curr), nu'(E_next)]
                p_hist = np.concatenate(([1.], nu_primes[i, :k+1], [current_primes[i]]))
                
                spline = CubicHermiteSpline(e_hist, v_hist, p_hist)
                temp_splines.append(spline)
            
            # 3. Compute RHS Integrals
            residuals = np.zeros(n_types)
            
            for i in range(n_types):
                rhs_sum = 0.0
                '''
                use Simpson's intergration
                for j in range(n_types):
                    limit = M[i, j] * E_next
                    
                    # Integration Setup
                    # We integrate T from 0 to M*E_next.
                    # Singularity at T=0. We use a linspace starting slightly > 0
                    # For accuracy, use more points or quad. Simpson on grid is okay.
                    # Grid for integration:
                    n_int_points = 100 
                    t_points = np.linspace(1e-5, limit, n_int_points)
                    
                    # Evaluate Integrand
                    integrand_vals = integrand_kernel(
                        t_points, E_next, temp_splines[j], temp_splines[i], i, j, U[i, j]
                    )
                    
                    integral = simpson(integrand_vals, x=t_points)
                    rhs_sum += G[i, j] * integral
                
                
                '''
                # Use quad to avoid error in root()
                nu_i_E_curr = current_vals[i] 
        
                for j in range(n_types):
                    limit = M[i, j] * E_next
            
                    # Use quad for robust integration starting from T=0
                    integral, err = quad(
                        integrand_kernel_for_quad, 
                        0.0,
                        limit,
                        # Arguments: E_next, nu_j, nu_i, nu_i(E_next), U_ij
                        args=(E_next, temp_splines[j], temp_splines[i], nu_i_E_curr, U[i, j])
                    )
            
                    # CRITICAL CHECK: Ensure integral is finite
                    if not np.isfinite(integral):
                        # Return NaN array to the root solver to indicate failure
                        print(f"QUAD FAILED: Integral for i={i}, j={j} is {integral} at E={E_next:.2e}")
                        return np.full(n_types, np.nan) 
                
                    rhs_sum += G[i, j] * integral

                # Residual = E * nu'(E) - RHS_integral
                # Eq (8) in docx: E * nu' = Sum ...
                residuals[i] = E_next * current_primes[i] - rhs_sum
                
            return residuals

        # Predictor: Linear extrapolation of slope or just use previous slope
        guess_slope = nu_primes[:, k]
        
        # Corrector: Solve for slopes that satisfy the integral eq
        sol = root(residual_func, guess_slope, method='hybr', tol=1e-4)
        
        if not sol.success:
            print(f"Warning: Convergence failed at E={E_next:.2e} eV")
            
        # Update State
        final_primes = sol.x
        nu_primes[:, k+1] = final_primes
        for i in range(n_types):
            nu_vals[i, k+1] = nu_vals[i, k] + (h/2.0) * (nu_primes[i, k] + final_primes[i])
            
        if k % 20 == 0:
            print(f"Step {k}: E={E_next:.2e}, nu/E={nu_vals[:, k+1]/E_next}")

    return E_grid, nu_vals

# ==========================================
# 4. Example Usage: Al2O3 and Y2O3
# ==========================================

if __name__ == "__main__":
    # Define Material: Al2O3
    # Elements: [(Z, A), ...]
    '''
    elements = [get_atomic_data('Al'), get_atomic_data('O')]
    stoichiometry = [2, 3] # Al2O3
    labels = ["Al", "O"]
    
    print("Solving for Al2O3...")
    E, nu = solve_polyatomic_damage_energy("Al2O3", elements, stoichiometry, E_max=1e7)
    
    # Plotting results
    # plt.figure(figsize=(10, 6))
    
    # Calculate damage efficiency nu(E)/E
    for i in range(len(elements)):
        efficiency = nu[i] / E
        plt.semilogx(E, efficiency, label=f"{labels[i]} in Al2O3", linewidth=2)
        
    plt.xlabel("PKA energy (eV)")
    plt.ylabel("Damage Efficiency v(E)/E")
    plt.title("Damage Efficiency in Polyatomic Material (Al2O3)")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.show()
    '''

    ### import and plot Coulter and Parkin's data for comparison
    ### Coulter_Parkin_Ref.py include their digital data
    import Coulter_Parkin_Ref
    
    y2o3_data = Coulter_Parkin_Ref.coulter_parkin_y2o3()
    plt.scatter(y2o3_data[:,0], y2o3_data[:,1], label=r'Y in Y$_2$O$_3$ Coulter')
    plt.scatter(y2o3_data[:,0], y2o3_data[:,3], label=r'O in Y$_2$O$_3$ Coulter')
    
    elements = [get_atomic_data('Y'), get_atomic_data('O')]
    stoichiometry = [2, 3] # Y2O3
    labels = ["Y", "O"]
    
    print("Solving for Y2O3...")
    E, nu = solve_polyatomic_damage_energy("Y2O3", elements, stoichiometry, E_max=1e7)
    # data_to_save = np.column_stack([E] + nu)
    # np.savetxt('Y2O3.dat', data_to_save, fmt='%.3e')
    # Plotting results
    
    
    # Calculate damage efficiency nu(E)/E
    for i in range(len(elements)):
        efficiency = nu[i] / E
        plt.semilogx(E, efficiency, label=f"{labels[i]} in Y$_2$O$_3$", linewidth=2)

    plt.xlim(1e1, 1e7)
    plt.xlabel("PKA energy (eV)")
    plt.ylabel(r"Damage efficiency $\nu$(E)/E")
    #plt.title("Damage Efficiency in Polyatomic Material (Y2O3)")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    #plt.savefig('Y2O3.png', dpi=150)
    plt.show()
