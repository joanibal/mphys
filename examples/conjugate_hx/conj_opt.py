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


parser.add_argument('--level', default='L4', choices=['L1', 'L2', 'L3', 'L4'])
# parser.add_argument('--aero', default='adflow', choices=['adflow', 'vlm'])
# parser.add_argument('--struct', default='tacs', choices=['tacs', 'modal'])
# parser.add_argument('--xfer', default='meld', choices=['meld', 'rlt'])
parser.add_argument('--nmodes', default=15)
parser.add_argument('--input_dir', default = './INPUT')
parser.add_argument('--driver', default='scipy', choices=['scipy', 'snopt'])
parser.add_argument('--aitken', default=True)
args = parser.parse_args()

# ------------------------ Helper Functions --------------------------

# def plotCoords(coords, DVGeo_foil, coords_new=None):
#     """plots a veiw of the airfoil coordinates and FFD"""

#     plt.plot(coords[:, 0], coords[:, 1], 'o--')
#     plt.plot(DVGeo_foil.FFD.coef[:, 0], DVGeo_foil.FFD.coef[:, 1], 'ko')
#     plt.plot(DVGeo_foil.children[0].FFD.coef[:, 0],
#              DVGeo_foil.children[0].FFD.coef[:, 1], 'co')
#     plt.plot(DVGeo_foil.children[1].FFD.coef[:, 0],
#              DVGeo_foil.children[1].FFD.coef[:, 1], 'ro')

#     if np.any(coords_new):
#         plt.plot(coords_new[:, 0], coords_new[:, 1], 'm-*')

#     plt.axis('equal')
#     # plt.title(MPI.COMM_WORLD.rank)
#     plt.show()

ffd_file = './ffds/nacelle_15_8_fixed.xyz'


def scale_sections(val, geo):
    for i in range(nSpanwise):
            geo.scale['centerline'].coef[i] = val[i]



nSpanwise = 15



def getDVGeo(DVGeo, ffd_file):
    """ returns the DVGeo for the deployed condition """
    from .ffd_utils import readFFDFile, getSections

    
    coords, ffd_size = readFFDFile(ffd_file)
    sections = getSections(coords, ffd_size, section_idx=0)


    centroid = np.mean(sections, axis=1)
    c0 = pySpline.Curve(X=centroid, k=2)

    DVGeo.addRefAxis('centerline', curve=c0, axis='y')


    DVGeo.addGeoDVGlobal('scale_sections', np.ones(ffd_size[0])*1.0,
                        scale_sections,
                        lower=np.ones(ffd_size[0])*0.5,
                        upper=np.ones(ffd_size[0])*3,
                        scale=1)


    return DVGeo




# geoFuncsDict = {
#     'scale_sections': {
#         'func': scale_sections,
#         'bound': [1., 10.],
#         'scale': 1e0
#     }
# }

class Top(om.Group):

    def setup(self):

        ################################################################################
        # AERO
        ################################################################################


        aero_options = {
            # I/O Parameters
            'gridFile':'./meshes/array_temp/nacelle_' + args.level + '.cgns',
            'outputDirectory':args.output_dir,
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
        ################################################################################
        # TRANSFER
        ################################################################################

        xfer_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}


        ap_runup = AeroProblem(name='fc_runup_{}'.format(args.level),
                        V=32, #m/s
                        T=273 + 60, # kelvin
                        P=93e3, # pa
                        areaRef=1.0,  #m^2
                        chordRef=1.0, #m^2
                        evalFuncs=[ 'cd', 'heatflux', 'havg'],
                        alpha=0.0, beta=0.00,
                        xRef=0.0, yRef=0.0, zRef=0.0)


        group = 'isothermalwall'
        BCVar = 'Temperature'
        bc_data = CFDSolver.getBCData()
        print(MPI.COMM_WORLD.rank, bc_data.getBCArraysFlatData(BCVar, familyGroup=group))
        ap_runup.setBCVar('Temperature', bc_data.getBCArraysFlatData(BCVar, familyGroup=group), group)
        ap_runup.addDV('Temperature', familyGroup=group, name='wall_temp', units='K')
        

        ap_cruise = AeroProblem(name='fc_cruise',
                 V=54, #m/s (105 kts)
                #  altitude= 1828.8, # m (6000 ft) 
                 T = 276,  #kelvin (based on standard atmosphere at 6000 ft)
                 P = 81.2e3, # pa (based on standard atmosphere at 6000 ft)
                 areaRef=0.1615**2*np.pi/4,  #m^2 0.1615 is diameter of motor
                 chordRef=0.9144, #m
                 evalFuncs=[ 'cd'],
                 alpha=4.0, beta=0.00,
                 xRef=0.0, yRef=0.0, zRef=0.0)


        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc for DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('alpha', val=3.725)
        dvs.add_output('scale_sections', shape_by_conn=True)
        self.connect('scale_sections',['conj_hxfer.scale_sections'])



        conj_analysis = Analysis()
        conj_analysis.add_subsystem('aero_mesh',AdflowMesh(solver_options=aero_options, family_groups=['allWalls', 'allIsothermalWalls']))
        conj_analysis.add_subsystem('geo_allwalls', DVGeoComp(ffd_file=ffd_file), promotes=['scale_sections'])
        conj_analysis.add_subsystem('geo_heatedwalls', DVGeoComp(ffd_file=ffd_file, setup_dvgeo=getDVGeo), promotes=['scale_sections'])
        conj_analysis.connect('aero_mesh.Xsurf_allWalls', ['geo_allwalls.pts'])
        conj_analysis.connect('aero_mesh.Xsurf_allIsothermalWalls', ['geo_heatedwalls.pts'])

        conj_analysis.connect('geo_allwalls.deformed_pts', ['x_a'])
        conj_analysis.connect('geo_heatedwalls.deformed_pts', ['cond.x_a'])
        # conj_analysis.connect('aero_mesh.Xsurf_allIsothermalWalls', ['cond.x_a'])



        conj_analysis.add_subsystem('aero',
                          AdflowGroup(aero_problem = ap_cruise, 
                           solver_options = aero_options, 
                           group_options = {
                               'mesh': False,
                               'deformer': True,
                               'funcs':True
                           }),
                           promotes=['x_a'])
        mda = Analysis()

        mda.add_subsystem('conv',
                          AdflowGroup(aero_problem = ap_runup, 
                           solver_options = aero_options, 
                           group_options = {
                               'mesh': False,
                               'deformer': True,
                               'heatxfer':True,
                               'funcs':False
                           }),
                           promotes=['wall_temp', 'x_a'])
                           # I'm explicitly promoteing the aero problem var, but it would
                           # be better to do this through a tag
                           
        mda.add_subsystem('cond',
                          ConductionNodal())
        
        mda.connect('cond.T_surf', ['wall_temp'])
        mda.connect('conv.heatflux', ['cond.heatflux'])


        mda.nonlinear_solver=om.NonlinearBlockGS(maxiter=100)
        mda.linear_solver = om.LinearBlockGS(maxiter=100)
        mda.nonlinear_solver.options['iprint']=2
        # solver options
        mda.nonlinear_solver.options['use_aitken'] = args.aitken
        mda.nonlinear_solver.options['atol'] = 1e-4
        mda.nonlinear_solver.options['rtol'] = 1e-20
        mda.linear_solver.options['atol'] = 1e-5
        mda.linear_solver.options['rtol'] = 1e-5
        mda.linear_solver.options['iprint'] = 2


        conj_analysis.add_subsystem('mda', mda, promotes=['*'])


        # conj_analysis.add_subsystem('aero_funcs',AdflowFunctions(solver_options=aero_options, aero_problem=ap0), promotes=['*'])
        # conj_analysis.add_subsystem('struct_funcs',TacsFunctions(solver_options=struct_options), promotes=['*'])
        # conj_analysis.add_subsystem('struct_mass',TacsMass(solver_options=struct_options), promotes=['*'])

        self.add_subsystem('conj_hxfer', conj_analysis)


    # def configure(self):
    #     self.geo.DVGeo = getDVGeo(self.geo.DVGeo, ffd_file)


    #     # create geometric DV setup
    #     points = self.mp_group.mphys_add_coordinate_input()
    #     # add these points to the geometry object
    #     self.geo.nom_add_point_dict(points)



################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()


if args.driver == 'scipy':
    #prob.driver = om.ScipyOptimizeDriver(debug_print=['ln_cons','nl_cons','objs','totals'])
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-3
    prob.driver.options['disp'] = True

    prob.driver.recording_options['includes'] = ['*']
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.options['debug_print'] =['desvars']

elif args.driver == 'snopt':
    
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.options['debug_print'] =['desvars', 'ln_cons', 'nl_cons', 'objs']
    prob.driver.opt_settings ={
        'Major feasibility tolerance': 1e-4, #1e-4,
        'Major optimality tolerance': 1e-4, #1e-8,
        'Verify level': 0,
        'Major iterations limit':600,
        'Minor iterations limit':1000000,
        'Iterations limit':1500000,
        #'Nonderivative linesearch':None,
        'Major step limit': 0.1,
        #'Function precision':1.0e-8,
        # 'Difference interval':1.0e-6,
        'Hessian full memory':None,
        'Hessian frequency' : 200,
        'Hessian updates': 200,
        #'Linesearch tolerance':0.99,
        'Print file':'%s/SNOPT_print.out'%args.output_dir,
        'Summary file': '%s/summary_SNOPT.out'%args.output_dir,
        'Problem Type':'Minimize',
        #'New superbasics limit':500,
        'Penalty parameter': 1.0, # 1.0,
    }

# recorder = om.SqliteRecorder("%s/cases.sql"%args.output_dir)
# prob.driver.add_recorder(recorder)
prob.model.add_design_var('scale_sections',lower=0.7, upper=1.6)
prob.model.add_objective('conj_hxfer.aero.cd',ref=1e-3)


# prob.model.add_constraint('conj_hxfer.conv.heatflux',ref=500.0,equals=500.0)
# prob.model._set_subsys_connection_errors(False)
recorder = om.SqliteRecorder(args.output_dir + '/hist_opt.db', record_viewer_data=False)
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_responses'] = True
prob.driver.recording_options['record_objectives'] = True
prob.setup(mode='rev')

# ss = prob.get_val('scale_sections', get_remote=True)
# prob.set_val('scale_sections', np.arange(ss.size)*0.1 + 0.7 )


om.n2(prob, show_browser=False, outfile='%s/mphys_conj_opt.html'%(args.output_dir))
if args.task == 'analysis':
    prob.run_model()
    # prob.model.list_outputs()
    # if MPI.COMM_WORLD.rank == 0:
    #     print("Scenario 0")
    #     if args.aero == 'adflow':
    #         print('cl =',prob['aerostruct.cl'])
    #         print('cd =',prob['aerostruct.cd'])
    #     else:
    #         print('cl =',prob['aerostuct.solver_group.aero.forces.CL'])
    #         print('cd =',prob['aerostuct.solver_group.aero.forces.CD'])

elif args.task == 'opt':
    # prob.driver.recording_options['record_constraints'] = True

    prob.run_driver()
    prob.model.list_outputs()

elif args.task == 'movie':

    # we will set the iteration limit of the cfd to super low and get tons of output files
    prob.model.mp_group.aero_builder.solver.setOption('ncycles', 2)
    # also adjust ank settings so we converge slower
    prob.model.mp_group.aero_builder.solver.setOption('anksecondordswitchtol', 1e-16)
    prob.model.mp_group.aero_builder.solver.setOption('ankcoupledswitchtol', 1e-16)
    prob.model.mp_group.aero_builder.solver.setOption('nsubiterturb', 10)
    prob.model.mp_group.aero_builder.solver.setOption('ankcfllimit', 50.0)

    prob.run_model()