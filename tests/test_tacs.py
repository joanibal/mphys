""" tests the componets included in the MPHYS wrapper for TACS"""
import unittest
import openmdao.api as om
import numpy as np
from mphys.mphys_tacs import TacsMesh, TacsSolver, TacsFunctions, TacsMass, PrescribedLoad, TACSGroup

from tacs import elements, constitutive, TACS, functions

@unittest.skip('')
class TestTACSSubsys(unittest.TestCase):
    N_Procs = 2

    def setUp(self):
        """ keep the options used to init the tacs solver options here """


        ################################################################################
        # Tacs solver pieces
        ################################################################################
        def add_elements(mesh):
            rho = 2500.0  # density, kg/m^3
            E = 70.0e9 # elastic modulus, Pa
            nu = 0.3 # poisson's ratio
            kcorr = 5.0 / 6.0 # shear correction factor
            ys = 350e6  # yield stress, Pa
            thickness = 0.020
            min_thickness = 0.00
            max_thickness = 1.00

            num_components = mesh.getNumComponents()
            for i in range(num_components):
                descript = mesh.getElementDescript(i)
                stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                            min_thickness, max_thickness)
                element = None
                if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
                    element = elements.MITCShell(2,stiff,component_num=i)
                mesh.setElement(i, element)

            ndof = 6
            ndv = num_components

            return ndof, ndv

        def get_funcs(tacs):
            ks_weight = 50.0
            return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

        def forcer_function(x_s,ndof):
            # apply uniform z load
            f_s = np.zeros(int(x_s.size/3)*ndof)
            f_s[2::ndof] = 100.0
            return f_s

        def f5_writer(tacs):
            flag = (TACS.ToFH5.NODES |
                    TACS.ToFH5.DISPLACEMENTS |
                    TACS.ToFH5.STRAINS |
                    TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile('ucrm.f5')

        self.solver_options = {'add_elements': add_elements,
                    'mesh_file'   : 'wingbox.bdf',
                    'get_funcs'   : get_funcs,
                    'load_function': forcer_function,
                    'f5_writer'   : f5_writer}

        self.top =  om.Group()

    def test_mesh(self):
        self.top.add_subsystem('mesh',TacsMesh(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

    def test_solver(self):
        self.top.add_subsystem('solver',TacsSolver(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()


    def test_functions(self):
        self.top.add_subsystem('funcs',TacsFunctions(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

    def test_mass(self):
        self.top.add_subsystem('mass',TacsMass(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

    def test_PrescribedLoad(self):
        self.top.add_subsystem('PrescribedLoad',PrescribedLoad(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

@unittest.skip('')
class TestTACSGroup(unittest.TestCase):
    N_Procs = 2

    def setUp(self):
        """ keep the options used to init the tacs solver options here """


        ################################################################################
        # Tacs solver pieces
        ################################################################################
        def add_elements(mesh):
            rho = 2500.0  # density, kg/m^3
            E = 70.0e9 # elastic modulus, Pa
            nu = 0.3 # poisson's ratio
            kcorr = 5.0 / 6.0 # shear correction factor
            ys = 350e6  # yield stress, Pa
            thickness = 0.020
            min_thickness = 0.00
            max_thickness = 1.00

            num_components = mesh.getNumComponents()
            for i in range(num_components):
                descript = mesh.getElementDescript(i)
                stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                            min_thickness, max_thickness)
                element = None
                if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
                    element = elements.MITCShell(2,stiff,component_num=i)
                mesh.setElement(i, element)

            ndof = 6
            ndv = num_components

            return ndof, ndv

        def get_funcs(tacs):
            ks_weight = 50.0
            return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

        def forcer_function(x_s,ndof):
            # apply uniform z load
            f_s = np.zeros(int(x_s.size/3)*ndof)
            f_s[2::ndof] = 100.0
            return f_s

        def f5_writer(tacs):
            flag = (TACS.ToFH5.NODES |
                    TACS.ToFH5.DISPLACEMENTS |
                    TACS.ToFH5.STRAINS |
                    TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile('ucrm.f5')

        self.solver_options = {'add_elements': add_elements,
                    'mesh_file'   : 'wingbox.bdf',
                    'get_funcs'   : get_funcs,
                    'load_function': forcer_function,
                    'f5_writer'   : f5_writer}

        self.top =  om.Group()

    def test_group(self):

        self.top.add_subsystem('struct',TACSGroup(solver_options=self.solver_options,
                                              group_options={
                                                  'loads':True,
                                              }) )


        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

class TestTACSGroup(unittest.TestCase):
    N_Procs = 2

    def setUp(self):
        """ keep the options used to init the tacs solver options here """


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
            flag = (TACS.ToFH5.NODES |
                    TACS.ToFH5.DISPLACEMENTS |
                    TACS.ToFH5.STRAINS |
                    TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile('%s/wingbox.f5'%args.output_dir)


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
       
        self.top =  om.Group()

    def test_group(self):

        self.top.add_subsystem('struct',TACSGroup(solver_options=self.solver_options,
                                              group_options={
                                                  'loads':True,
                                              }) )


        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()


if __name__ == '__main__':
    unittest.main()