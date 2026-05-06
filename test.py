
import PolyDamE

labels = ["Ni","Cr", "Fe", "Nb", "Ti"]
stoichiometry = [74, 14.9, 7.3, 1., 2.3]

elements = [PolyDamE.get_atomic_data(el) for el in labels]
E, nu = PolyDamE.solve_polyatomic_damage_energy("InconelX750", \
                                elements, stoichiometry, E_max=1e7)

print ('================ Calculations finished ================')
print ('Results are given in the file named {material_name}.dat')
