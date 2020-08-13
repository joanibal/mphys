from conduction_models import ConductionNodal
import openmdao.api as om
import numpy as np 
from adflow import ADFLOW


if False:
    aero_options = {
        # I/O Parameters
        'gridFile':'./meshes/array_temp/nacelle_' + 'L4' + '.cgns',
        'outputDirectory':'./output',
        'monitorvariables':['resrho','resturb','cl','cd', 'heatflux'],
        'surfacevariables': ['cp','vx', 'vy', 'vz', 'mach', 'heatflux', 'temp'],
        # 'isovariables': ['shock'],
        'isoSurface':{'shock':1}, #,'vx':-0.0001},
        'writeTecplotSurfaceSolution':False,
        'writevolumesolution':False,
        # 'writesurfacesolution':False,
        'liftindex':3,

        # Physics Parameters
        'equationType':'RANS',

        # Solver Parameters
        'smoother':'dadi',
        'CFL':1.5,
        'MGCycle':'sg',
        'MGStartLevel':-1,

        # ANK Solver Parameters
        'useANKSolver':True,
        'nsubiterturb': 5,
        'anksecondordswitchtol':1e-4,
        'ankcoupledswitchtol': 1e-6,
        'ankinnerpreconits':2,
        'ankouterpreconits':1,
        'anklinresmax': 0.1,
        'infchangecorrection':True,
        'ankcfllimit': 1e4,

        # NK Solver Parameters
        # 'useNKSolver':True,
        'nkswitchtol':1e-4,

        # Termination Criteria
        'L2Convergence':1e-13,
        'L2ConvergenceCoarse':1e-2,
        'L2ConvergenceRel': 1e-3,
        'nCycles':3000,
        'adjointl2convergencerel': 1e-3,


    }
    CFDSolver = ADFLOW(options=aero_options)



    prob = om.Problem()


    dvs = prob.model.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])


    x_a = CFDSolver.getSurfaceCoordinates(groupName='allIsothermalWalls')
    dvs.add_output('x_a', val=x_a)

    heatflux = np.linspace(-200, -300, x_a.size//3)

    dvs.add_output('heatflux', val=heatflux)


    prob.model.add_subsystem('cond', ConductionNodal(), promotes=['*'])

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    data = prob.check_partials()



if True:
    prob = om.Problem()


# x1_error = data['comp']['y', 'x1']['abs error']

# print(x1_error.forward)

# x2_error = data['comp']['y', 'x2']['rel error']

# print(x2_error.forward)