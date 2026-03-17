import numpy as np

# ====================================================================
# Data and Functions for Coulter and Parkin (1980) Tables
# ====================================================================

# --- 1. Table 1: Analytic Fit Coefficients ---
# Format: {'Material': {'Atom': [C1, C2, C3]}}
COULTER_COEFFICIENTS = {
    'AL2O3': {
        'AL': [1.03736e-1, 5.91664e-5, 3.88700e-6],
        'O': [1.10761e-1, 1.12891e-4, 9.25824e-6],
    },
    'CAO': {
        'CA': [1.02321e-1, 3.21217e-5, 1.64816e-6],
        'O': [1.15138e-1, 1.08821e-4, 7.82439e-6],
    },
    'MGO': {
        'MG': [1.04809e-1, 6.59569e-5, 4.55073e-6],
        'O': [1.10459e-1, 1.11748e-4, 9.21318e-6],
    },
    'NBTI': {
        'NB': [8.39468e-2, 8.59845e-6, 3.34242e-7],
        'TI': [9.20061e-2, 1.96243e-5, 1.00050e-7],
    },
    'SI3N4': {
        'SI': [1.05558e-1, 5.59129e-5, 3.49171e-6],
        'N': [1.14772e-1, 1.36210e-4, 1.14617e-5],
    },
    'UC': {
        'U': [9.48584e-2, 5.46293e-6, 1.41433e-9],
        'C': [2.19182e-1, 2.70502e-4, 7.64238e-6],
    },
    'UO2': {
        'U': [9.53888e-2, 8.42709e-6, -1.23147e-8],
        'O': [1.64118e-1, 1.80911e-4, 5.35033e-6],
    },
    'Y2O3': {
        'Y': [9.98448e-2, 1.48893e-5, 3.55057e-7],
        'O': [1.31749e-1, 1.25650e-4, 6.90102e-6],
    },
    'TAO': {
        'TA': [9.57247e-2, 6.34122e-6, 6.43278e-8],
        'O': [1.67358e-1, 1.51191e-4, 5.15949e-6],
    },
    'MGAL2O4': {
        'MG': [1.05092e-1, 6.62556e-5, 4.56955e-6],
        'AL': [1.03782e-1, 5.89465e-5, 3.88665e-6],
        'O': [1.10766e-1, 1.12529e-4, 9.25109e-6],
    },
}

def coulter_parkin_nu2E_fit(E, material, atom):
    """
    Calculates the damage energy nu(E) using the analytic fit from Coulter and Parkin (Table 1).
    
    nu(E) = E / (1 + C1*E^0.15 + C2*E^0.75 + C3*E)
    
    Args:
        E (float or array): Energy in eV.
        material (str): The compound name (e.g., 'UO2', 'AL2O3').
        atom (str): The atom whose nu(E) is being calculated (e.g., 'U', 'O', 'AL').
        
    Returns:
        float or array: Damage energy nu(E) in eV.
    """
    material = material.upper()
    atom = atom.upper()
    
    try:
        C1, C2, C3 = COULTER_COEFFICIENTS[material][atom]
    except KeyError:
        raise ValueError(f"Coefficients not found for {atom} in {material}. Check spelling.")

    E_arr = np.atleast_1d(E)
    
    denominator = 1.0 + C1 * E_arr**0.15 + C2 * E_arr**0.75 + C3 * E_arr
    nu_E = 1. / denominator
    
    return nu_E.item() if np.isscalar(E) else nu_E


# --- 2. Table 2: Al2O3 Tabulated Data ---

def coulter_parkin_al2o3():
    """
    Returns the raw tabulated damage efficiency (nu/E) data for Al2O3 (Table 2).
    
    Columns are:
    [Energy (eV), Al in Al2O3, Al in Al (Self), O in Al2O3, O in O (Self)]
    """
    return np.array([
        [1e1,   0.878, 0.889, 0.868, 0.876],
        [1e2,   0.830, 0.842, 0.817, 0.824],
        [1e3,   0.766, 0.779, 0.745, 0.751],
        [1e4,   0.664, 0.680, 0.613, 0.615],
        [1e5,   0.441, 0.463, 0.318, 0.310],
        [1e6,   0.132, 0.144, 0.0682, 0.0644],
        [1e7,   0.0196, 0.0218, 0.0088, 0.0082],
    ]) 

# --- 3. Table 3: Y2O3 Tabulated Data ---

def coulter_parkin_y2o3():
    """
    Returns the raw tabulated damage efficiency (nu/E) data for Y2O3 (Table 3).
    
    Columns are:
    [Energy (eV), Y in Y2O3, Y in Y (Self), O in Y2O3, O in O (Self)]
    """
    return np.array([
        [1e1,   0.886, 0.914, 0.846, 0.876],
        [1e2,   0.842, 0.876, 0.790, 0.824],
        [1e3,   0.784, 0.826, 0.712, 0.751],
        [1e4,   0.708, 0.759, 0.583, 0.615],
        [1e5,   0.593, 0.659, 0.321, 0.310],
        [1e6,   0.384, 0.456, 0.0776, 0.0644],
        [1e7,   0.120, 0.150, 0.0107, 0.0082],
    ]) 


# --- 4. Table 6: U_xZr_(1-x)C Tabulated Data ---

# Data for Table 6, structured based on the primary knock-on atom.
# E values are in scientific notation (e.g., 1.12e1)
# Column structure: [E, Atom in Self, Atom in x=0.02, Atom in x=0.50, Atom in x=0.98]

U_DATA_TABLE_6 = np.array([
    [1.12e1, 0.921, 0.894, 0.889, 0.886],
    [1.07e2, 0.890, 0.855, 0.848, 0.846],
    [1.02e3, 0.850, 0.803, 0.796, 0.793],
    [1.15e4, 0.791, 0.732, 0.722, 0.720],
    [1.10e5, 0.719, 0.639, 0.631, 0.631],
    [1.05e6, 0.608, 0.498, 0.501, 0.510],
    [1.00e7, 0.388, 0.280, 0.297, 0.314]
])

ZR_DATA_TABLE_6 = np.array([
    [1.12e1, 0.904, 0.878, 0.870, 0.863],
    [1.07e2, 0.867, 0.835, 0.824, 0.815],
    [1.02e3, 0.818, 0.777, 0.764, 0.754],
    [1.15e4, 0.747, 0.696, 0.680, 0.670],
    [1.10e5, 0.652, 0.582, 0.570, 0.564],
    [1.05e6, 0.452, 0.387, 0.388, 0.394],
    [1.00e7, 0.154, 0.128, 0.137, 0.146]
])

C_DATA_TABLE_6 = np.array([
    [1.12e1, 0.859, 0.820, 0.797, 0.770],
    [1.07e2, 0.806, 0.758, 0.730, 0.697],
    [1.02e3, 0.726, 0.672, 0.638, 0.600],
    [1.15e4, 0.538, 0.507, 0.472, 0.436],
    [1.10e5, 0.208, 0.235, 0.222, 0.211],
    [1.05e6, 0.036, 0.050, 0.051, 0.052],
    [1.00e7, 0.005, 0.007, 0.007, 0.008]
])


def coulter_parkin_uxzrc(primary_atom):
    """
    Returns the raw tabulated damage efficiency (nu/E) data for U_xZr_(1-x)C (Table 6).
    
    Args:
        primary_atom (str): The primary knock-on atom ('U', 'ZR', or 'C').
        
    Columns are structured as:
    [Energy (eV), nu/E (Atom in Self), nu/E (Atom in U_xZrC, x=0.02), 
     nu/E (Atom in U_xZrC, x=0.50), nu/E (Atom in U_xZrC, x=0.98)]
    """
    # primary_atom = primary_atom.upper()
    
    if primary_atom == 'U':
        return U_DATA_TABLE_6
    elif primary_atom == 'Zr':
        return ZR_DATA_TABLE_6
    elif primary_atom == 'C':
        return C_DATA_TABLE_6
    else:
        raise ValueError("Primary atom must be 'U', 'ZR', or 'C' for Table 6 data.")


if __name__ == '__main__':
    print("--- Coulter and Parkin Data Functions Loaded ---")
    
    # Example 1: Using the Table 1 analytic fit function
    E_test = 5e5 # 500 keV
    nu_U_UO2 = coulter_parkin_nu2E_fit(E_test, 'UO2', 'U')
    print(f"\nExample 1 (Table 1 Fit): nu(5e5 eV) for U in UO2 = {nu_U_UO2:.2e} eV")
    
    # Example 2: Using the Table 2 tabulated data
    al2o3_data = coulter_parkin_al2o3()
    E_1e4 = al2o3_data[3, 0] # Energy 1e4 eV
    nu_over_E_O_in_Al2O3 = al2o3_data[3, 3]
    print(f"\nExample 2 (Table 2 Data): nu/E for O in Al2O3 at {E_1e4:.0e} eV = {nu_over_E_O_in_Al2O3}")
    
    # Example 3: Using the Table 6 tabulated data
    u_x050_data = coulter_parkin_uxzrc('U')
    E_1e5 = u_x050_data[4, 0] # Energy 1.10e5 eV
    nu_over_E_U_in_x050 = u_x050_data[4, 3] # Column 3 is x=0.50
    print(f"\nExample 3 (Table 6 Data): nu/E for U in U(0.5)Zr(0.5)C at {E_1e5:.2e} eV = {nu_over_E_U_in_x050}")
