from __future__ import division, print_function
import numpy as np

import openmdao.api as om
from tacs import TACS,functions

from .base_classes import SolverObjectBasedSystem
from .analysis import Analysis

from tacs import TACS
from .base_classes import  ObjBuilder, SysBuilder



class TacsObjs():
    def __init__(self, mesh, assembler, mat, pc, ksp):
        self.mesh = mesh
        self.assembler = assembler
        self.mat = mat
        self.pc = pc
        self.ksp = ksp


class TacsObjsBuilder(ObjBuilder):
    def __init__(self):
        super().__init__(TacsObjs)

    def build_obj(self, comm):
        mesh = TACS.MeshLoader(comm)
        mesh.scanBDFFile(self.options['mesh_file'])
        ndof, ndv = self.options['add_elements'](mesh)
        assembler = mesh.createTACS(ndof)

        try:
            mat = assembler.createFEMat()
        except AttributeError:
            mat = assembler.createSchurMat()



        pc = TACS.Pc(mat)

        # TODO these should be set as options with a default
        subspace = 100
        restarts = 2
        ksp = TACS.KSM(mat, pc, subspace, restarts)


        return TacsObjs(mesh, assembler, mat, pc, ksp)




class TacsMesh(om.ExplicitComponent):
    """
    Component to read the initial mesh coordinates with TACS

    """
    def initialize(self):
        # self.options.declare('get_tacs', default = None, desc='function to get tacs')
        self.options.declare('check_partials', default=False)
        self.options.declare('solver_options')

        self.options['distributed'] = True



        self.objBuilders = [TacsObjsBuilder()]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for b in self.objBuilders:
            b.options = self.options['solver_options']


    def setup(self):
        for b in self.objBuilders:
            if b.obj is None:
                b.obj = b.build_obj(self.comm)

        self.assem = self.objBuilders[0].obj.assembler

        # create some TACS bvecs that will be needed later
        self.xpts  = self.assem.createNodeVec()
        self.assem.getNodes(self.xpts)

        # OpenMDAO setup
        xpts_arr = self.xpts.getArray().flatten()
        # node_size  =     xpts_arr.size
        self.add_output('x_s0', val=xpts_arr, desc='structural node coordinates')

        if 'surface_mapping' in self.options['solver_options']:
            xpts_surf_arr, self.mapping = self.options['solver_options']['surface_mapping'](xpts_arr)
            xpts_surf_arr = xpts_surf_arr.flatten()

            self.add_output('x_surf_s0', val=xpts_surf_arr, desc='sub set of structural node coordinates given by mapping')



    def mphys_add_coordinate_input(self):
        local_size  = self.xpts.getArray().size
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_s0_points', shape=node_size, src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')

        # return the promoted name and coordinates
        return 'x_s0_points', self.xpts.getArray()

    def compute(self,inputs,outputs):
        if 'x_s0_points' in inputs:
            outputs['x_s0'] = inputs['x_s0_points']
        # else:
        #     outputs['x_s0'] = self.xpts.getArray()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'x_s0_points' in d_inputs:
                d_outputs['x_s0'] += d_inputs['x_s0_points']
        elif mode == 'rev':
            if 'x_s0_points' in d_inputs:
                d_inputs['x_s0_points'] += d_outputs['x_s0']

class TacsSolver(om.ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - User will provide a tacs_solver_setup function that gives some pieces
          required for the tacs solver
          => tacs, mat, pc, gmres, struct_ndv = tacs_solver_setup(comm)
        - The TACS steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('solver_options')
        self.options.declare('check_partials', default=False)

        self.options['distributed'] = True



        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.psi_s = None
        self.x_save = None

        self.transposed = False
        self.check_partials = False

        self.old_dvs = None

        self.objBuilders = [TacsObjsBuilder()]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for b in self.objBuilders:
            b.options = self.options['solver_options']


    def setup(self):

        options = self.options['solver_options']
        self.check_partials = self.options['check_partials']

        for b in self.objBuilders:
            if b.obj is None:
                b.obj = b.build_obj(self.comm)



        if 'f5_writer' in  self.options['solver_options']:
            self.f5_writer = self.options['solver_options']['f5_writer']
        else:
            self.f5_writer = None


        # TACS assembler setup
        self.mesh      = self.objBuilders[0].obj.mesh
        self.tacs      = self.objBuilders[0].obj.assembler
        self.mat       = self.objBuilders[0].obj.mat
        self.pc        = self.objBuilders[0].obj.pc
        self.gmres     = self.objBuilders[0].obj.ksp

        self.ndv = self.mesh.getNumComponents()

        # create some TACS bvecs that will be needed later
        self.res        = self.tacs.createVec()
        self.force      = self.tacs.createVec()
        self.ans        = self.tacs.createVec()
        self.struct_rhs = self.tacs.createVec()
        self.psi_s      = self.tacs.createVec()
        self.xpt_sens   = self.tacs.createNodeVec()

        # OpenMDAO setup

        state_size = self.ans.getArray().size
        node_size  = self.xpt_sens.getArray().size
        self.ndof = int(state_size/(node_size/3))

        s_list = self.comm.allgather(state_size)
        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        print(irank,n1, n2 )





        # inputs
        self.add_input('x_s0', shape=node_size , src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')

        if 'Conduction' in self.options['solver_options']:
            self.heat       = self.tacs.createVec()

            self.xpts  = self.tacs.createNodeVec()
            self.tacs.getNodes(self.xpts)
            xpts_array = self.xpts.getArray()
            xpts_hot_surf, self.mapping = options['surface_mapping'](xpts_array)
            # xpts_hot_surf = xpts_array[self.mapping]

            n_nodes_hot_surf = xpts_hot_surf.size//3




            s_list = self.comm.allgather(n_nodes_hot_surf)
            irank  = self.comm.rank

            s1 = np.sum(s_list[:irank])
            s2 = np.sum(s_list[:irank+1])

            self.add_input('heat_xfer',       shape=n_nodes_hot_surf, src_indices=np.arange(s1, s2, dtype=int), desc='structural load vector')

            # outputs
            print('conduction temp_cond', n_nodes_hot_surf)
            self.add_output('temp',      shape=n_nodes_hot_surf, val = np.ones(n_nodes_hot_surf)*300,desc='temperature vector')

        else:
            self.add_input('f_s', shape=state_size, src_indices=np.arange(s1, s2, dtype=int), desc='structural load vector')
            self.add_input('dv_struct', shape=self.ndv, desc='tacs design variables')
            # outputs
            # its important that we set this to zero since this displacement value is used for the first iteration of the aero
            self.add_output('u_s', shape=state_size, val = np.zeros(state_size),desc='structural state vector')



    def _need_update(self,inputs):


        if self.old_dvs is None:
            self.old_dvs = inputs['dv_struct'].copy()
            return True

        for dv, dv_old in zip(inputs['dv_struct'],self.old_dvs):
            if np.abs(dv - dv_old) > 1e-7:
                self.old_dvs = inputs['dv_struct'].copy()
                return True

        return False

    def _update_internal(self,inputs,outputs=None):
        if not 'dv_struct' in inputs:
            pc     = self.pc
            alpha = 1.0
            beta  = 0.0
            gamma = 0.0

            xpts = self.tacs.createNodeVec()
            self.tacs.getNodes(xpts)
            xpts_array = xpts.getArray()
            xpts_array[:] = inputs['x_s0']
            self.tacs.setNodes(xpts)

            res = self.tacs.createVec()
            res_array = res.getArray()
            res_array[:] = 0.0

            self.tacs.assembleJacobian(alpha,beta,gamma,res,self.mat)
            pc.factor()

            return

        if self._need_update(inputs):
            self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

            xpts = self.tacs.createNodeVec()
            self.tacs.getNodes(xpts)
            xpts_array = xpts.getArray()
            xpts_array[:] = inputs['x_s0']
            self.tacs.setNodes(xpts)

            pc     = self.pc
            alpha = 1.0
            beta  = 0.0
            gamma = 0.0

            xpts = self.tacs.createNodeVec()
            self.tacs.getNodes(xpts)
            xpts_array = xpts.getArray()
            xpts_array[:] = inputs['x_s0']
            self.tacs.setNodes(xpts)

            res = self.tacs.createVec()
            res_array = res.getArray()
            res_array[:] = 0.0

            self.tacs.assembleJacobian(alpha,beta,gamma,res,self.mat)
            pc.factor()

        if outputs is not None:
            ans = self.ans
            ans_array = ans.getArray()
            ans_array[:] = outputs['u_s']
            self.tacs.applyBCs(ans)

            self.tacs.setVariables(ans)

    def apply_nonlinear(self, inputs, outputs, residuals):
        tacs = self.tacs
        res  = self.res
        ans  = self.ans

        self._update_internal(inputs,outputs)

        res_array = res.getArray()
        res_array[:] = 0.0

        # K * u
        tacs.assembleRes(res)

        # Add the external loads
        res_array[:] -= inputs['f_s']

        # Apply BCs to the residual (forces)
        tacs.applyBCs(res)

        residuals['u_s'][:] = res_array[:]
        import ipdb; ipdb.set_trace()

    def solve_nonlinear(self, inputs, outputs):
        tacs   = self.tacs
        force  = self.force
        ans    = self.ans
        pc     = self.pc
        gmres  = self.gmres



        pc     = self.pc
        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        res = self.tacs.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self._update_internal(inputs)

        if 'Conduction' in self.options['solver_options']:
            heat = self.heat

            heat.zeroEntries()
            heat_array = heat.getArray()
            # may need to do mapping here
            heat_array[self.mapping] = inputs['heat_xfer']


            self.tacs.setBCs(heat)

            gmres.solve(heat, ans)
            ans_array = ans.getArray()
            tacs.setVariables(ans)


            ans_array = ans.getArray()

            outputs['temp'] = ans_array[self.mapping]
            print(outputs['temp'])

            if self.f5_writer is not None:
                self.f5_writer(self.tacs)

        else:
            # solve the linear system
            force_array = force.getArray()
            force_array[:] = inputs['f_s']
            tacs.applyBCs(force)

            gmres.solve(force, ans)
            ans_array = ans.getArray()
            outputs['u_s'] = ans_array[:]
            tacs.setVariables(ans)

    def solve_linear(self,d_outputs,d_residuals,mode):
        if mode == 'fwd':
            if self.check_partials:
                print ('solver fwd')
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            tacs = self.tacs
            gmres = self.gmres

            # if nonsymmetric, we need to form the transpose Jacobian
            #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
            #    alpha = 1.0
            #    beta  = 0.0
            #    gamma = 0.0

            #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
            #    pc.factor()
            #    self.transposed=True

            res = self.res
            res_array = res.getArray()




            if 'Conduction' in self.options['solver_options']:


                res_array[self.mapping] = d_outputs['temp']

                before = res_array.copy()
                tacs.applyBCs(res)
                after = res_array.copy()
                psi_s = self.psi_s
                gmres.solve(res,psi_s)

                psi_s_array = psi_s.getArray()
                # tacs.applyBCs(psi_s)
                tacs.applyBCs(psi_s)


                # print('inputs seed', d_outputs['temp'])
                print('after - before', np.array(after - before,dtype=np.float64)[self.mapping])
                # print()
                d_residuals['temp'] = psi_s_array.copy()[self.mapping]
                d_residuals['temp'] -= np.array(after - before,dtype=np.float64)[self.mapping]
                # print(d_residuals['temp'])

            else:


                res_array[:] = d_outputs['u_s']
                before = res_array.copy()
                tacs.applyBCs(res)
                after = res_array.copy()
                psi_s = self.psi_s
                gmres.solve(res,psi_s)
                psi_s_array = psi_s.getArray()
                tacs.applyBCs(psi_s)
                d_residuals['u_s'] = psi_s_array.copy()
                d_residuals['u_s'] -= np.array(after - before,dtype=np.float64)


                # # solve the linear system
                # force_array = force.getArray()
                # force_array[:] = inputs['f_s']
                # tacs.applyBCs(force)

                # gmres.solve(force, ans)
                # ans_array = ans.getArray()
                # outputs['u_s'] = ans_array[:]
                # tacs.setVariables(ans)



    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        self._update_internal(inputs,outputs)
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'temp' in d_residuals:
                tacs = self.tacs

                res  = self.res
                res_array = res.getArray()

                ans  = self.ans
                ans_array = ans.getArray()

                psi = tacs.createVec()
                psi_array = psi.getArray()
                psi_array[self.mapping] = d_residuals['temp']

                before = psi_array.copy()
                tacs.applyBCs(psi)
                after = psi_array.copy()

                if 'temp' in d_outputs:

                    ans_array[self.mapping] = outputs['temp']
                    tacs.applyBCs(ans)
                    tacs.setVariables(ans)

                    # if nonsymmetric, we need to form the transpose Jacobian
                    #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
                    #    alpha = 1.0
                    #    beta  = 0.0
                    #    gamma = 0.0
                    #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
                    #    pc.factor()
                    #    self.transposed=True

                    res_array[:] = 0.0
                    self.mat.mult(psi,res)
                    # tacs.applyBCs(res)

                    d_outputs['temp'] += np.array(res_array[self.mapping],dtype=float)
                    d_outputs['temp'] -= np.array(after - before,dtype=np.float64)

                if 'heat_xfer' in d_inputs:
                    d_inputs['heat_xfer'] -= np.array(psi_array[self.mapping],dtype=float)

                if 'x_s0' in d_inputs:
                    # xpt_sens = self.xpt_sens

                    xpt_sens   = tacs.createNodeVec()
                    xpt_sens_array = xpt_sens.getArray()


                    tacs.addAdjointResXptSensProducts([psi], [xpt_sens])

                    d_inputs['x_s0'] += np.array(xpt_sens_array[:],dtype=float)


            if 'u_s' in d_residuals:
                tacs = self.tacs

                res  = self.res
                res_array = res.getArray()

                ans  = self.ans
                ans_array = ans.getArray()

                psi = tacs.createVec()
                psi_array = psi.getArray()
                psi_array[:] = d_residuals['u_s'][:]

                before = psi_array.copy()
                tacs.applyBCs(psi)
                after = psi_array.copy()

                if 'u_s' in d_outputs:

                    ans_array[:] = outputs['u_s']
                    tacs.applyBCs(ans)
                    tacs.setVariables(ans)

                    # if nonsymmetric, we need to form the transpose Jacobian
                    #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
                    #    alpha = 1.0
                    #    beta  = 0.0
                    #    gamma = 0.0
                    #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
                    #    pc.factor()
                    #    self.transposed=True

                    res_array[:] = 0.0

                    self.mat.mult(psi,res)
                    # tacs.applyBCs(res)

                    d_outputs['u_s'] += np.array(res_array[:],dtype=float)
                    d_outputs['u_s'] -= np.array(after - before,dtype=np.float64)

                if 'f_s' in d_inputs:
                    d_inputs['f_s'] -= np.array(psi_array[:],dtype=float)

                if 'x_s0' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()

                    tacs.evalAdjointResXptSensProduct(psi, xpt_sens)

                    d_inputs['x_s0'] += np.array(xpt_sens_array[:],dtype=float)

                if 'dv_struct' in d_inputs:
                    adj_res_product  = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                    self.tacs.evalAdjointResProduct(psi, adj_res_product)

                    # TACS has already done a parallel sum (mpi allreduce) so
                    # only add the product on one rank
                    if self.comm.rank == 0:
                        d_inputs['dv_struct'] +=  np.array(adj_res_product,dtype=float)

    def _design_vector_changed(self,x):
        if self.x_save is None:
            self.x_save = x.copy()
            return True
        elif not np.allclose(x,self.x_save,rtol=1e-10,atol=1e-10):
            self.x_save = x.copy()
            return True
        else:
            return False


class TacsFunctions(om.ExplicitComponent):
    """
    Component to compute TACS functions

    Assumptions:
        - User will provide a tacs_func_setup function that will set up a list of functions
          => func_list, tacs, struct_ndv = tacs_func_setup(comm)
    """
    def initialize(self):
        # self.options.declare('struct_solver')
        self.options.declare('check_partials', default=False)
        self.options.declare('solver_options')
        self.objBuilders = [TacsObjsBuilder()]

        self.options['distributed'] = True


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for b in self.objBuilders:
            b.options = self.options['solver_options']


    def setup(self):
        self.check_partials = self.options['check_partials']

        for b in self.objBuilders:
            if b.obj is None:
                b.obj = b.build_obj(self.comm)

        # TACS assembler setup
        self.mesh      = self.objBuilders[0].obj.mesh
        self.tacs      = self.objBuilders[0].obj.assembler

        self.ndv = ndv = self.mesh.getNumComponents()


        get_funcs = self.options['solver_options']['get_funcs']


        if 'f5_writer' in  self.options['solver_options']:
            self.f5_writer = self.options['solver_options']['f5_writer']
        else:
            self.f5_writer = None




        # TACS assembler setup
        self.func_list = get_funcs(self.tacs)

        self.ans = self.tacs.createVec()
        state_size = self.ans.getArray().size

        self.xpt_sens = self.tacs.createNodeVec()
        node_size = self.xpt_sens.getArray().size

        print(node_size, self.tacs.getNumNodes())

        s_list = self.comm.allgather(state_size)
        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        # OpenMDAO part of setup
        # TODO move the dv_struct to an external call where we add the DVs
        self.add_input('dv_struct', shape=ndv,                                                    desc='tacs design variables')
        self.add_input('x_s0',      shape=node_size,  src_indices=np.arange(n1, n2, dtype=int),   desc='structural node coordinates')
        self.add_input('u_s',       shape=state_size, src_indices=np.arange(s1, s2, dtype=int),   desc='structural state vector')

        # Remove the mass function from the func list if it is there
        # since it is not dependent on the structural state
        func_no_mass = []
        for i,func in enumerate(self.func_list):
            if not isinstance(func,functions.StructuralMass):
                func_no_mass.append(func)

        self.func_list = func_no_mass
        if len(self.func_list) > 0:
            self.add_output('f_struct', shape=len(self.func_list), desc='structural function values')

            # declare the partials
            #self.declare_partials('f_struct',['dv_struct','x_s0','u_s'])

    def _update_internal(self,inputs):
        self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        mat    = self.tacs.createFEMat()
        pc     = TACS.Pc(mat)
        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        res = self.tacs.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self.tacs.assembleJacobian(alpha,beta,gamma,res,mat)
        pc.factor()

        ans = self.ans
        ans_array = ans.getArray()
        ans_array[:] = inputs['u_s']
        self.tacs.applyBCs(ans)

        self.tacs.setVariables(ans)

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'f_struct' in outputs:
            print('f_struct',outputs['f_struct'])
            outputs['f_struct'] = self.tacs.evalFunctions(self.func_list)

        if self.f5_writer is not None:
            self.f5_writer(self.tacs)

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if self.check_partials:
                self._update_internal(inputs)

            if 'f_struct' in d_outputs:
                for ifunc, func in enumerate(self.func_list):
                    self.tacs.evalFunctions([func])
                    if 'dv_struct' in d_inputs:
                        dvsens = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                        self.tacs.evalDVSens(func, dvsens)

                        d_inputs['dv_struct'][:] += np.array(dvsens,dtype=float) * d_outputs['f_struct'][ifunc]

                    if 'x_s0' in d_inputs:
                        xpt_sens = self.xpt_sens
                        xpt_sens_array = xpt_sens.getArray()
                        self.tacs.evalXptSens(func, xpt_sens)

                        d_inputs['x_s0'][:] += np.array(xpt_sens_array,dtype=float) * d_outputs['f_struct'][ifunc]

                    if 'u_s' in d_inputs:
                        prod = self.tacs.createVec()
                        self.tacs.evalSVSens(func,prod)
                        prod_array = prod.getArray()

                        d_inputs['u_s'][:] += np.array(prod_array,dtype=float) * d_outputs['f_struct'][ifunc]

class TacsMass(om.ExplicitComponent):
    """
    Component to compute TACS mass

    Assumptions:
        - User will provide a tacs_func_setup function that will set up a list of functions
          => func_list, tacs, struct_ndv = tacs_func_setup(comm)
    """
    def initialize(self):
        self.options.declare('solver_options')
        self.options.declare('check_partials', default=False)

        self.options['distributed'] = True

        self.ans = None
        self.tacs = None

        self.mass = False

        self.objBuilders = [TacsObjsBuilder()]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for b in self.objBuilders:
            b.options = self.options['solver_options']


    def setup(self):
        self.check_partials = self.options['check_partials']

        for b in self.objBuilders:
            if b.obj is None:
                b.obj = b.build_obj(self.comm)

        # TACS assembler setup
        self.mesh      = self.objBuilders[0].obj.mesh
        self.tacs      = self.objBuilders[0].obj.assembler

        self.ndv = ndv = self.mesh.getNumComponents()

        # crea

        self.xpt_sens = self.tacs.createNodeVec()
        node_size = self.xpt_sens.getArray().size

        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        # OpenMDAO part of setup
        self.add_input('dv_struct', shape=ndv,                                                    desc='tacs design variables')
        self.add_input('x_s0',      shape=node_size,  src_indices=np.arange(n1, n2, dtype=int),   desc='structural node coordinates')

        self.add_output('mass', 0.0, desc = 'structural mass')
        #self.declare_partials('mass',['dv_struct','x_s0'])

    def _update_internal(self,inputs):
        self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'mass' in outputs:
            func = functions.StructuralMass(self.tacs)
            outputs['mass'] = self.tacs.evalFunctions([func])

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if self.check_partials:
                self._update_internal(inputs)
            if 'mass' in d_outputs:
                func = functions.StructuralMass(self.tacs)
                if 'dv_struct' in d_inputs:
                    size = d_inputs['dv_struct'].size
                    dvsens = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                    self.tacs.evalDVSens(func, dvsens)

                    d_inputs['dv_struct'] += np.array(dvsens,dtype=float) * d_outputs['mass']

                if 'x_s0' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()
                    self.tacs.evalXptSens(func, xpt_sens)

                    d_inputs['x_s0'] += np.array(xpt_sens_array,dtype=float) * d_outputs['mass']


class PrescribedLoad(om.ExplicitComponent):
    """
    Prescribe a load to tacs

    Assumptions:
        - User will provide a load_function prescribes the loads
          => load = load_function(x_s0,ndof)

    """
    def initialize(self):
        self.options.declare('solver_options')
        self.options.declare('check_partials', default=True)


        self.options['distributed'] = True

        self.ndof = 0
        self.objBuilders = [TacsObjsBuilder()]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for b in self.objBuilders:
            b.options = self.options['solver_options']


    def setup(self):

        if not 'load_function' in self.options['solver_options']:
            raise KeyError('`load_function` must be supplied in the solver options dictionary'+ \
                            ' to use the PrescribedLoad componet')
        self.check_partials = self.options['check_partials']

        for b in self.objBuilders:
            if b.obj is None:
                b.obj = b.build_obj(self.comm)

        # TACS assembler setup
        self.tacs      = self.objBuilders[0].obj.assembler

        # create some TACS vectors so we can see what size they are
        # TODO getting the node sizes should be easier than this...
        xpts  = self.tacs.createNodeVec()
        node_size = xpts.getArray().size

        tmp   = self.tacs.createVec()
        state_size = tmp.getArray().size
        self.ndof = int(state_size / ( node_size / 3 ))

        irank = self.comm.rank

        n_list = self.comm.allgather(node_size)
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        # OpenMDAO setup
        self.add_input('x_s0', shape=node_size, src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')
        self.add_output('f_s', shape=state_size, desc='structural load')

        #self.declare_partials('f_s','x_s0')

    def compute(self,inputs,outputs):
        load_function = self.options['solver_options']['load_function']
        outputs['f_s'] = load_function(inputs['x_s0'],self.ndof)






class TACSGroup(Analysis):
    def initialize(self):
        super().initialize()
        self.options.declare('solver_options')
        self.options.declare('group_options')
        self.options.declare('check_partials', default=False)



        self.group_options = {
            'mesh': False,
            'loads': False,
            'solver': True,
            'funcs': True,
            'mass': True,
        }

        self.group_components = {
            'mesh': TacsMesh,
            'loads': PrescribedLoad,
            'solver': TacsSolver,
            'funcs': TacsFunctions,
            'mass': TacsMass,
        }


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # set given options
        self.check_partials = self.options['check_partials']
        self.group_options.update(self.options['group_options'])

        solver_options = self.options['solver_options']

        # set the solver options on the solver objects
        # add required componets to the subsystem.
        print('============TACS================')
        for comp in self.group_components:
            if self.group_options[comp]:
                print(comp)
                self.add_subsystem(comp, self.group_components[comp](solver_options=solver_options,
                                                                    check_partials=self.check_partials),
                            promotes=['*']) # we can connect things implicitly through promotes
                                            # because we already know the inputs and outputs of each
                                            # components




