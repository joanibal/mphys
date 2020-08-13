import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om


# Instantiate your CaseReader
cr = om.CaseReader("output/hist_opt.db")
driver_cases = cr.list_cases('driver')

dv_x_values = []
dv_z_values = []
for i in range(len(driver_cases)):
    last_case = cr.get_case(driver_cases[i])
    design_vars = last_case.get_design_vars()
    if design_vars:
        dv_x_values.append(design_vars['x'])
        dv_z_values.append(design_vars['z'])

# Below is a short script to see the path the design variables took to convergence

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
ax1.plot(np.arange(len(dv_x_values)), np.array(dv_x_values))

ax1.set(xlabel='Iterations', ylabel='Design Var: X', title='Optimization History')
ax1.grid()

ax2.plot(np.arange(len(dv_z_values)), np.array(dv_z_values))

ax2.set(xlabel='Iterations', ylabel='Design Var: Z', title='Optimization History')
ax2.grid()