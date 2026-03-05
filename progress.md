1. Obtinere date treptele cu prop. solid Vega/Vega C/Vega E pentru a avea date cat mai precise. Datele vor fi cel mai probabil lipsite de scara numerica la dorinta ESA (ca in Agostino et al.)

2. Level set method (Fortran/Python)
    - implementarea metodei level set pentru propagarea pe normala a vitezei de ardere (multiple sectiuni de-a lungul batonului de propelant)

3. Determinarea si implementarea modelului corect Euler cvasi-1D necesar lucrarii. Cercetarea diverselor augmentari care pot fi adaugate + de ce sunt/nu sunt adaugate pentru rachetele alese.

4. Adaugarea arderii erozive. Lenoir Robillard pare cel mai bun, folosit inclusiv de Agostino. In functie de forma ecuatiilor Euler 1D, acest pas poate fi absorbit de pasul 3.

5. Calibrarea modelului pe un esantion de cateva trepte si testarea pe alt esantion pentru stabilirea factorior ce necesita corectie.

6. (mai tarziu in lucrare) Dezvoltarea modelului structural. Aplicarea lui pe rachetele deja existente

7. (mai tarziu in lucrare) Aplicarea modelului structural pe rachetele deja existente. Analiza diferentelor


Termen limita 1-5: 10 aprilie (15 aprilie termen limita pentru incarcarea abstractelor)

1. Date cerute: 
- propellant used - mass, geometry, chemical mix (enthalpy for constituents if available), burning temperature, burning rate properties, density, heat capacity of the propellant
- geometry of the propellant case - outer shell geometry (length, diameter, wall thickness), internal insulation material between propellant and case (thickness, distribution)
- nozzle geometry (throat cross sectional area, exit cross sectional area)
- flight profile - burning time, thrust developed over time (with corresponding ambient pressure over time), internal pressure over time if recorded, specific impulse (sea level or vacuum)
- burnt gasses estimation - heat capacity ratio (gamma), specific heat (Cp), thermal conductivity, gas viscosity (mu)


Measure-Command { & C:\Users\ionst\AppData\Local\Programs\Python\Python313\python.exe c:/Users/ionst/Documents/GitHub/Erosive-Burning-Euler-1D/Euler-1D/Riemann_solver.py } | Select-Object TotalSeconds