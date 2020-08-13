from mpi4py import MPI
import argparse
import numpy as np
import openmdao.api as om

# from mphys.mphys_multipoint import MPHYS_Multipoint

from mphys.mphys_adflow import *
from mphys.mphys_tacs import *


from adflow import ADFLOW

from mphys.mphys_meld import MELDThermal_heat_xfer_rate_xfer, MELDThermal_temp_xfer


# only try to import this so that people can run the script w/o mdolab code
try:
    from baseclasses import *
except:
    pass

# TACS is required regardless of the structural solver used
from tacs import elements, constitutive, functions
# from struct_dv_components import StructDvMapper, SmoothnessEvaluatorGrid, struct_comps

parser=argparse.ArgumentParser()
parser.add_argument('--task', default='analysis', choices=['analysis', 'opt', 'movie'])

# parser.add_argument("--output_dir", type=str, default=os.path.join(os.environ["DATAFOLDER"], 'nacelle_opt_array'))
parser.add_argument('--output_dir', default = './output')
parser.add_argument('--debug',   help='set debugging options', action='store_true')


# parser.add_argument('--aero', default='adflow', choices=['adflow', 'vlm'])
# parser.add_argument('--struct', default='tacs', choices=['tacs', 'modal'])
# parser.add_argument('--xfer', default='meld', choices=['meld', 'rlt'])
parser.add_argument('--input_dir', default = './INPUT')
parser.add_argument('--driver', default='scipy', choices=['scipy', 'snopt'])
parser.add_argument('--aitken', default=True)
args = parser.parse_args()





class Top(om.Group):

    def setup(self):

        ################################################################################
        # AERO
        ################################################################################


        aero_options = {
            # 'printTiming': False,

            # Common Parameters
            'gridFile': 'naca0012_hot_rans.cgns',
            'outputDirectory': './',
            # 'discretization': 'upwind',

            # 'oversetupdatemode': 'full',
            'volumevariables': ['temp'],
            'surfacevariables': ['cf', 'vx', 'vy', 'vz', 'temp', 'heattransfercoef', 'heatflux'],
            'monitorVariables':	['resturb', 'yplus', 'heatflux'],
            # Physics Parameters
            # 'equationType': 'laminar NS',
            'equationType': 'rans',
            # 'vis2':0.0,
            'liftIndex': 2,
            'CFL': 1.0,
            # 'smoother': 'DADI',
            # 'smoother': 'runge',

            'useANKSolver': True,
            'ANKswitchtol': 10e0,
            # 'ankcfllimit': 5e6,
            'anksecondordswitchtol': 5e-3,
            'ankcoupledswitchtol': 1e-7,
            # NK parameters
            'useNKSolver': True,
            'nkswitchtol': 1e-5,
            
            'rkreset': False,
            'nrkreset': 40,
            'MGCycle': 'sg',
            # 'MGStart': -1,
            # Convergence Parameters
            'L2Convergence': 1e-12,
            'nCycles': 1000,
            'nCyclesCoarse': 250,
            'ankcfllimit': 5e3,
            'nsubiterturb': 5,
            'ankphysicallstolturb': 0.99,
            'anknsubiterturb': 5,
            # 'ankuseturbdadi': False,
            'ankturbkspdebug': True,

            'storerindlayer': True,
            # Turbulence model
            'eddyvisinfratio': .210438,
            'useft2SA': False,
            'turbulenceproduction': 'vorticity',
            'useblockettes': False,

        }

        ################################################################################
        # TRANSFER
        ################################################################################

        xfer_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}


        # ap_runup = AeroProblem(name='fc_runup_{}'.format(args.level),
        #                 V=32, #m/s
        #                 T=273 + 60, # kelvin
        #                 P=93e3, # pa
        #                 areaRef=1.0,  #m^2
        #                 chordRef=1.0, #m^2
        #                 evalFuncs=[ 'cd', 'heatflux', 'havg'],
        #                 alpha=0.0, beta=0.00,
        #                 xRef=0.0, yRef=0.0, zRef=0.0)

        # atmospheric conditions
        temp_air = 273  # kelvin
        Pr = 0.72
        mu = 1.81e-5  # kg/(m * s)

        u_inf = 10  # m/s\
        p_inf = 101e3
        ap = AeroProblem(name='fc_conv', V=u_inf, T=temp_air,
                rho=1.225, areaRef=1.0, chordRef=1.0, alpha=10.0, beta=0,  evalFuncs=['cl', 'cd'])

        ap_runup = ap

        group = 'heated_wall'
        BCVar = 'Temperature'
        
        CFDSolver = ADFLOW(options=aero_options)
        bc_data = CFDSolver.getBCData()
        print(MPI.COMM_WORLD.rank, bc_data.getBCArraysFlatData(BCVar, familyGroup=group))
        ap_runup.setBCVar('Temperature', np.ones(bc_data.getBCArraysFlatData(BCVar, familyGroup=group).size)*300.0, group)
        ap_runup.addDV('Temperature', familyGroup=group, name='wall_temp', units='K')
        

        conv_sys = AdflowGroup(aero_problem = ap_runup, 
                    solver_options = aero_options, 
                    group_options = {
                        'mesh': False,
                        'deformer': True,
                        'heatxfer':True,
                        'funcs':False
                    })

        ################################################################################
        # STRUCT
        ################################################################################
        def add_elements(mesh):
            # Create the constitutvie propertes and model
            props = constitutive.MaterialProperties(kappa = 230)
            # con = constitutive.PlaneStressConstitutive(props)
            con = constitutive.SolidConstitutive(props)
            heat = elements.HeatConduction3D(con)

            # Create the basis class
            # quad_basis = elements.LinearQuadBasis()
            basis = elements.LinearHexaBasis()

            # Create the element
            element = elements.Element3D(heat, basis)


            # Loop over components, creating stiffness and element object for each
            num_components = mesh.getNumComponents()
            for i in range(num_components):
                descriptor = mesh.getElementDescript(i)
                print('Setting element with description %s'%(descriptor))
                mesh.setElement(i, element)


            ndof = heat.getVarsPerNode()

            return ndof, 0


        def get_surface_mapping(Xpts_array):


            Xpts_array = Xpts_array.reshape(len(Xpts_array)//3, 3)

            unique_x = set(Xpts_array[:,0])
            unique_x = list(unique_x)
            unique_x.sort()

            plate_surface = [] 
            mask = []
            lower_mask = []
            upper_mask = []

            for x in unique_x:
                mask_sec = np.where(Xpts_array[:, 0] == x)[0]
                

                # find min and max y points
                max_mask = np.where(Xpts_array[mask_sec,1] == np.max(Xpts_array[mask_sec, 1]))[0]
                min_mask = np.where(Xpts_array[mask_sec,1] == np.min(Xpts_array[mask_sec, 1]))[0]

                lower_mask.extend(mask_sec[min_mask])
                upper_mask.extend(mask_sec[max_mask])

                # mask.extend(mask_sec[min_mask], mask_sec[max_mask])
                
                # plate_surface.extend([lower_mask, upper_mask])
                # mapping.extend

            lower_mask = np.array(lower_mask)
            upper_mask = np.array(upper_mask)
            print(MPI.COMM_WORLD.rank, lower_mask)
            mask = np.concatenate((lower_mask, upper_mask))
            mapping = mask
            plate_surface = np.array(Xpts_array[mask])


            # plate_surface = []
            # mapping = []
            # for i in range(len(Xpts_array) // 3):

            #     # check if it's on the flow edge
            #     if Xpts_array[3*i+1] == 0.0:
            #         plate_surface.extend(Xpts_array[3*i:3*i+3])
            #         mapping.append(i)


            # plate_surface = np.array(plate_surface)
            return plate_surface, mapping

        def get_funcs(tacs):
            ks_weight = 50.0
            return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

        tacs_options = {
            'add_elements': add_elements,
            'mesh_file'   : 'flatplate.bdf',
            # 'get_funcs': get_funcs,
            'get_surface': get_surface_mapping,
        }

        def f5_writer(tacs):


            flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_STRAINS)
            

            f5 = TACS.ToFH5(tacs, TACS.SOLID_ELEMENT, flag)
            fid = 'tacs_iter_%04d.f5' %f5_writer.count
            f5.writeToFile(fid)
            print('tacs output written to ', fid)

            f5_writer.count += 1


        f5_writer.count = 0

        # common setup options
        struct_options = {
            'add_elements': add_elements,
            # 'get_funcs'   : get_funcs,
            'mesh_file'   : 'n0012_hexa.bdf',
            'f5_writer'   : f5_writer,
            'surface_mapping': get_surface_mapping,
            'Conduction': True
        }

        cond_sys = TACSGroup(solver_options=struct_options,
                               group_options={
                                   'loads':False,
                                   'mesh':False,
                                   'mass':False,
                                   'funcs':False
                               })
       

        ################################################################################
        # TRANSFER
        ################################################################################

        xfer_options = {'isym': 1,
                        'n': 10,
                        'beta': 0.5}

        temp_xfer_sys = MELDThermal_temp_xfer(solver_options = xfer_options)
        heatflux_xfer_sys = MELDThermal_heat_xfer_rate_xfer(solver_options = xfer_options)

        ################################################################################



        # ivc for DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('alpha', val=3.725)
        # self.connect('alpha', ['conj.alpha'])

   
        # create the multiphysics analysis group.

        conj_sys = Analysis()
        conj_sys.add_subsystem('struct_mesh',TacsMesh(solver_options=struct_options))
        conj_sys.add_subsystem('aero_mesh',AdflowMesh(solver_options=aero_options, family_groups=['allWalls', 'heated_wall']))


        mda = Analysis()
        mda.add_subsystem('conv', conv_sys, promotes=['q', 'x_g', 'x_a'])    

        mda.add_subsystem('heat_xfer_xfer', heatflux_xfer_sys, promotes=['x_conv0', 'x_cond0'])
        mda.connect('conv.heatflux', ['heat_xfer_xfer.heat_xfer_conv'])
        mda.connect('heat_xfer_xfer.heat_xfer_cond', ['cond.heat_xfer'])
        
        mda.add_subsystem('cond', cond_sys, promotes=['x_s0'])
        mda.connect('cond.temp', ['temp_xfer.temp_cond'])

        mda.add_subsystem('temp_xfer', temp_xfer_sys, promotes=['x_conv0', 'x_cond0'])
        mda.connect('temp_xfer.temp_conv', ['conv.wall_temp'])



        



        mda.nonlinear_solver=om.NonlinearBlockGS(maxiter=100)
        mda.linear_solver = om.LinearBlockGS(maxiter=100)
        # solver options
        mda.nonlinear_solver.options['use_aitken'] = True
        mda.nonlinear_solver.options['aitken_max_factor'] = 0.5
        mda.nonlinear_solver.options['aitken_min_factor'] = 0.05
        mda.nonlinear_solver.options['aitken_initial_factor'] = 0.2
        # mda.linear_solver.options['atol'] = 1e-1
        mda.linear_solver.options['rtol'] = 1e-1
        mda.linear_solver.options['iprint'] = 2


        conj_sys.add_subsystem('mda', mda, promotes=['*'])

        conj_sys.connect('aero_mesh.Xsurf_allWalls', ['x_a'])
        conj_sys.connect('aero_mesh.Xsurf_heated_wall', ['x_conv0'])
        conj_sys.connect('struct_mesh.x_s0', ['x_s0'])
        conj_sys.connect('struct_mesh.x_surf_s0', ['x_cond0'])
        # conj_sys.add_subsystem('aero_funcs',AdflowFunctions(solver_options=aero_options, aero_problem=ap0), promotes=['*'])
        # conj_sys.add_subsystem('struct_funcs',TacsFunctions(solver_options=struct_options), promotes=['*'])
        # conj_sys.add_subsystem('struct_mass',TacsMass(solver_options=struct_options), promotes=['*'])

        self.add_subsystem('conj', conj_sys)


   
################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile='mphys_n0012_conj.html')


if args.task == 'analysis':
    prob.run_model()
# prob.model.list_outputs()
# if MPI.COMM_WORLD.rank == 0:
#     print("Scenario 0")
#     print('cl =',prob['mp_group.s0.aero.funcs.cl'])
#     print('cd =',prob['mp_group.s0.aero.funcs.cd'])
