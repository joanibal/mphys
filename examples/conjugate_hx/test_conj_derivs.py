from mpi4py import MPI
import argparse
import numpy as np
import openmdao.api as om

# from mphys.mphys_multipoint import MPHYS_Multipoint

from mphys.mphys_adflow import *
from mphys.mphys_tacs import *
from mphys.dvgeo_component_configure import OM_DVGEOCOMP, DVGeoComp

from pyspline import pySpline

from adflow import ADFLOW

from  conduction_models import Conduction, ConductionNodal

# only try to import this so that people can run the script w/o mdolab code
try:
    from baseclasses import *
except:
    pass

# TACS is required regardless of the structural solver used
from tacs import elements, constitutive, functions
# from struct_dv_components import StructDvMapper, SmoothnessEvaluatorGrid, struct_comps

from ffd_utils import readFFDFile, getSections

from pprint import pprint


parser=argparse.ArgumentParser()
parser.add_argument('--task', default='analysis', choices=['analysis', 'opt', 'movie'])

# parser.add_argument("--output_dir", type=str, default=os.path.join(os.environ["DATAFOLDER"], 'nacelle_opt_array'))
parser.add_argument('--output_dir', default = './output')
parser.add_argument('--debug',   help='set debugging options', action='store_true')


parser.add_argument('--level', default='L5', choices=['L1', 'L2', 'L3', 'L4', 'L5'])
# parser.add_argument('--aero', default='adflow', choices=['adflow', 'vlm'])
# parser.add_argument('--struct', default='tacs', choices=['tacs', 'modal'])
# parser.add_argument('--xfer', default='meld', choices=['meld', 'rlt'])
parser.add_argument('--nmodes', default=15)
parser.add_argument('--input_dir', default = './INPUT')
parser.add_argument('--driver', default='scipy', choices=['scipy', 'snopt'])
parser.add_argument('--aitken', default=True)

parser.add_argument('--step_size', help='step size', type=float, default=5e-3)
parser.add_argument('--min_heatxfer', help='lower bound on heatflux (negated in func)', type=float, default=0)

args = parser.parse_args()


if False:
    aero_options = {
        # I/O Parameters
        'gridFile':'./meshes/array_temp/nacelle_' + 'L4' + '.cgns',
        'outputDirectory':'./output',
        'monitorvariables':['resrho','resturb','cl','cd'],
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


    x_a = CFDSolver.getSurfaceCoordinates(groupName='allIsothermalWalls').flatten()
    dvs.add_output('x_a', val=x_a)

    heatflux = np.linspace(-200, -300, x_a.size//3)

    dvs.add_output('heatflux', val=heatflux)


    prob.model.add_subsystem('cond', ConductionNodal(), promotes=['*'])

    prob.set_solver_print(level=0)

    prob.setup(force_alloc_complex=True)
    prob.run_model()

    data = prob.check_partials(method='cs')



if False:
    prob = om.Problem()


    aero_options = {
        # I/O Parameters
        'gridFile':'./meshes/array_temp/nacelle_' + args.level + '.cgns',
        'outputDirectory':args.output_dir,
        'monitorvariables':['resrho','resturb','cl','cd', 'totheattransfer'],
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
        'L2Convergence':1e-14,
        'L2ConvergenceCoarse':1e-2,
        'nCycles':4000,
        'adjointl2convergence': 1e-14,


    }

    ap_runup = AeroProblem(name='fc_runup_{}'.format(args.level),
                    V=32, #m/s
                    T=273 + 60, # kelvin
                    P=93e3, # pa
                    areaRef=1.0,  #m^2
                    chordRef=1.0, #m^2
                    evalFuncs=[ 'cd', 'totheattransfer', 'havg'],
                    alpha=0.0, beta=0.00,
                    xRef=0.0, yRef=0.0, zRef=0.0)

    group = 'isothermalwall'
    BCVar = 'Temperature'
    CFDSolver = ADFLOW(options=aero_options)

    bc_data = CFDSolver.getBCData()
    print(MPI.COMM_WORLD.rank, bc_data.getBCArraysFlatData(BCVar, familyGroup=group))
    ap_runup.setBCVar('Temperature', bc_data.getBCArraysFlatData(BCVar, familyGroup=group), group)
    ap_runup.addDV('Temperature', familyGroup=group, name='wall_temp', units='K')


    dvs = prob.model.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])


    x_a = CFDSolver.getSurfaceCoordinates(groupName='allWalls')
    x_a = x_a.flatten()
    dvs.add_output('wall_temp', val=bc_data.getBCArraysFlatData(BCVar, familyGroup=group))
    dvs.add_output('x_a', val=x_a)


    prob.model.add_subsystem('conv',
                        AdflowGroup(aero_problem = ap_runup,
                        solver_options = aero_options,
                        group_options = {
                            'mesh': False,
                            'deformer': True,
                            'heatxfer':True,
                            'funcs':False
                        }),
                        promotes=['x_a', 'wall_temp'])



    prob.set_solver_print(level=2)

    prob.model.add_design_var('x_a',  lower=0.7, upper=1.6)
    prob.model.add_design_var('wall_temp') # ,indices=[15])
    # prob.model.add_objective('totheattransfer',ref=1e2)
    prob.model.add_constraint('conv.heatflux', indices=[0], lower=-1*args.min_heatxfer*1.5,  upper=-1*args.min_heatxfer, scaler=1e2)

    prob.setup(mode='rev')
    # om.n2(prob, show_browser=False, outfile='test.html')
    prob.run_model()
    # quit()
    print('----------------strating check totals--------------')
    data = prob.check_totals(step=1e-6, form='forward', step_calc='rel')




if True:
    prob = om.Problem()


    aero_options = {
        # I/O Parameters
        'gridFile':'./meshes/array_temp/nacelle_' + args.level + '.cgns',
        'outputDirectory':args.output_dir,
        'monitorvariables':['resrho','resturb','cl','cd', 'totheattransfer'],
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
        'L2Convergence':1e-14,
        'L2ConvergenceCoarse':1e-2,
        'nCycles':4000,
        'adjointl2convergence': 1e-14,


    }

    ap_runup = AeroProblem(name='fc_runup_{}'.format(args.level),
                    V=32, #m/s
                    T=273 + 60, # kelvin
                    P=93e3, # pa
                    areaRef=1.0,  #m^2
                    chordRef=1.0, #m^2
                    evalFuncs=[ 'cd', 'totheattransfer', 'havg'],
                    alpha=0.0, beta=0.00,
                    xRef=0.0, yRef=0.0, zRef=0.0)

    group = 'isothermalwall'
    BCVar = 'Temperature'
    CFDSolver = ADFLOW(options=aero_options)

    bc_data = CFDSolver.getBCData()
    print(MPI.COMM_WORLD.rank, bc_data.getBCArraysFlatData(BCVar, familyGroup=group))
    ap_runup.setBCVar('Temperature', bc_data.getBCArraysFlatData(BCVar, familyGroup=group), group)
    ap_runup.addDV('Temperature', familyGroup=group, name='wall_temp', units='K')


    dvs = prob.model.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])


    x_a = CFDSolver.getSurfaceCoordinates(groupName='allWalls').flatten()
    x_a_hot = CFDSolver.getSurfaceCoordinates(groupName='allIsothermalWalls').flatten()
    # dvs.add_output('wall_temp', val=bc_data.getBCArraysFlatData(BCVar, familyGroup=group))
    dvs.add_output('x_a', val=x_a)
    dvs.add_output('x_a_hot', val=x_a_hot)


    mda = Analysis()

    mda.add_subsystem('cond',
                        ConductionNodal())

    mda.add_subsystem('conv',
                        AdflowGroup(aero_problem = ap_runup,
                        solver_options = aero_options,
                        group_options = {
                            'mesh': False,
                            'deformer': True,
                            'heatxfer':True,
                            'funcs':True
                        }),
                        promotes=['wall_temp', 'x_a', 'totheattransfer'])
                    #    promotes=['wall_temp', 'x_a'])
                        # I'm explicitly promoteing the aero problem var, but it would
                        # be better to do this through a tag


    mda.connect('cond.T_surf', ['wall_temp'])
    mda.connect('conv.heatflux', ['cond.heatflux'])
    mda.nonlinear_solver=om.NonlinearBlockGS(maxiter=20)
    mda.nonlinear_solver.options['use_aitken'] = args.aitken
    mda.nonlinear_solver.options['atol'] = 1e-20
    mda.nonlinear_solver.options['rtol'] = 1e-10
    mda.nonlinear_solver.options['iprint']=2

    # mda.linear_solver = om.PETScKrylov()
    mda.linear_solver = om.LinearBlockGS(maxiter=20)
    # mda.linear_solver.options['use_aitken'] = args.aitken

    # mda.linear_solver.options['atol'] = 1e-4
    mda.nonlinear_solver.options['atol'] = 1e-20
    mda.linear_solver.options['rtol'] = 1e-10
    mda.linear_solver.options['iprint'] = 2
    prob.model.add_subsystem('mda', mda, promotes=['*'])

    prob.model.connect('x_a_hot', ['cond.x_a'])

    # prob.model.add_subsystem('conv',
    #                     AdflowGroup(aero_problem = ap_runup,
    #                     solver_options = aero_options,
    #                     group_options = {
    #                         'mesh': False,
    #                         'deformer': True,
    #                         'heatxfer':True,
    #                         'funcs':True
    #                     }),
    #                     promotes=['x_a', 'wall_temp', 'totheattransfer'])



    prob.set_solver_print(level=2)

    prob.model.add_design_var('x_a',indices=[0], lower=0.7, upper=1.6)
    prob.model.add_objective('totheattransfer',ref=1e2)
    # prob.model.add_constraint('conj_hxfer.totheattransfer', lower=-1*args.min_heatxfer*1.5,  upper=-1*args.min_heatxfer, scaler=1e2)


    prob.setup()
    om.n2(prob, show_browser=False, outfile='test.html')
    prob.run_model()


    data = prob.check_totals()


# x1_error = data['comp']['y', 'x1']['abs error']

# print(x1_error.forward)

# x2_error = data['comp']['y', 'x2']['rel error']

# print(x2_error.forward)