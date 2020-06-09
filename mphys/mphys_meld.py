import numpy as np
import openmdao.api as om
from funtofem import TransferScheme

class MELD_disp_xfer(om.ExplicitComponent):
    """
    Component to perform displacement transfer using MELD
    """
    def initialize(self):
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.meld = None
        self.initialized_meld = False

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.check_partials = False

    def setup(self):

        self.check_partials= self.options['check_partials']

        self.meld = TransferScheme.pyMELD(self.comm,
                                          self.comm, 0,
                                          self.comm, 0,
                                          self.options['isym'],
                                          self.options['n'],
                                          self.options['beta'])



        # inputs
        self.add_input('x_s0', shape_by_connect = True, src_indices_by_connect = True, desc='initial structural node coordinates')
        self.add_input('x_a0', shape_by_connect = True, src_indices_by_connect = True, desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  determine_shape = True, src_indices_by_connect = True, desc='structural node displacements')

        # outputs
        self.add_output('u_a', determine_shape = True, desc='aerodynamic surface displacements')

    def determine_shape(self, var_meta_data):

        self.set_variable_shape('u_s', (var_meta_data['x_s0'].size//3))
    
        self.set_variable_shape('u_a', (var_meta_data['x_a0'].shape))
    

    def compute(self, inputs, outputs):
        x_s0 = np.array(inputs['x_s0'],dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs['x_a0'],dtype=TransferScheme.dtype)
        u_a  = np.array(outputs['u_a'],dtype=TransferScheme.dtype)

        u_s  = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs['u_s'][i::self.struct_ndof]

        self.meld.setStructNodes(x_s0)
        self.meld.setAeroNodes(x_a0)

        if not self.initialized_meld:
            self.meld.initialize()
            self.initialized_meld = True

        self.meld.transferDisps(u_s,u_a)

        outputs['u_a'] = u_a

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The MELD residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """
        if mode == 'fwd':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    d_in = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    for i in range(3):
                        d_in[i::3] = d_inputs['u_s'][i::self.struct_ndof]
                    prod = np.zeros(self.aero_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDduS(d_in,prod)
                    d_outputs['u_a'] -= np.array(prod,dtype=float)

                if 'x_a0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('forward mode requested but not implemented')

                if 'x_s0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_a' in d_outputs:
                du_a = np.array(d_outputs['u_a'],dtype=TransferScheme.dtype)
                if 'u_s' in d_inputs:
                    # du_a/du_s^T * psi = - dD/du_s^T psi
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDduSTrans(du_a,prod)
                    for i in range(3):
                        d_inputs['u_s'][i::self.struct_ndof] -= np.array(prod[i::3],dtype=np.float64)

                # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                if 'x_a0' in d_inputs:
                    prod = np.zeros(d_inputs['x_a0'].size,dtype=TransferScheme.dtype)
                    self.meld.applydDdxA0(du_a,prod)
                    d_inputs['x_a0'] -= np.array(prod,dtype=float)

                if 'x_s0' in d_inputs:
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDdxS0(du_a,prod)
                    d_inputs['x_s0'] -= np.array(prod,dtype=float)

class MELD_load_xfer(om.ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):
        self.options.declare('struct_ndof')

        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.meld = None
        self.initialized_meld = False

        self.struct_ndof = None

        self.check_partials = False

    def setup(self):
        # get the transfer scheme object
        self.meld = self.options['xfer_object']

        self.check_partials= self.options['check_partials']

        self.meld = TransferScheme.pyMELD(self.comm,
                                          self.comm, 0,
                                          self.comm, 0,
                                          self.options['isym'],
                                          self.options['n'],
                                          self.options['beta'])


        # inputs
        self.add_input('x_s0', shape_by_connect = True, src_indices_by_connect = True, desc='initial structural node coordinates')
        self.add_input('x_a0', shape_by_connect = True, src_indices_by_connect = True, desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  determine_shape = True, src_indices_by_connect = True, desc='structural node displacements')
        self.add_input('f_a',  determine_shape = True, src_indices_by_connect = True, desc='aerodynamic force vector')

        # outputs
        self.add_output('f_s', determine_shape = True, desc='structural force vector')

    def determine_shape(self, var_meta_data):

        self.set_variable_shape(['u_s', 'f_s'], (var_meta_data['x_s0'].size//3*self.struct_ndof))
        self.set_variable_shape(['f_a'], var_meta_data['x_a0'].shape)
    

        # partials
        #self.declare_partials('f_s',['x_s0','x_a0','u_s','f_a'])

    def compute(self, inputs, outputs):
        u_s  = np.zeros(self.struct_nnodes*3)
        for i in range(3):
            u_s[i::3] = inputs['u_s'][i::self.struct_ndof]

        f_a =  np.array(inputs['f_a'],dtype=TransferScheme.dtype)
        f_s = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)

        if self.check_partials:
            x_s0 = np.array(inputs['x_s0'],dtype=TransferScheme.dtype)
            x_a0 = np.array(inputs['x_a0'],dtype=TransferScheme.dtype)
            self.meld.setStructNodes(x_s0)
            self.meld.setAeroNodes(x_a0)
            #TODO meld needs a set state rather requiring transferDisps to update the internal state
            u_s  = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs['u_s'][i::self.struct_ndof]
            u_a = np.zeros(inputs['f_a'].size,dtype=TransferScheme.dtype)
            self.meld.transferDisps(u_s,u_a)

        self.meld.transferLoads(f_a,f_s)

        outputs['f_s'][:] = 0.0
        for i in range(3):
            outputs['f_s'][i::self.struct_ndof] = f_s[i::3]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            f_s = g(f_a,u_s,x_a0,x_s0)
        The MELD internal residual is defined as:
            L = f_s - g(f_a,u_s,x_a0,x_s0)
        So explicit partials below for f_s are negative partials of L
        """


        if mode == 'fwd':
            if 'f_s' in d_outputs:
                if 'u_s' in d_inputs:
                    d_in = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    for i in range(3):
                        d_in[i::3] = d_inputs['u_s'][i::self.struct_ndof]
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydLduS(d_in,prod)
                    for i in range(3):
                        d_outputs['f_s'][i::self.struct_ndof] -= np.array(prod[i::3],dtype=float)

                if 'f_a' in d_inputs:
                    # df_s/df_a psi = - dL/df_a * psi = -dD/du_s^T * psi
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    df_a = np.array(d_inputs['f_a'],dtype=TransferScheme.dtype)
                    self.meld.applydDduSTrans(df_a,prod)
                    for i in range(3):
                        d_outputs['f_s'][i::self.struct_ndof] -= np.array(prod[i::3],dtype=float)

                if 'x_a0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('forward mode requested but not implemented')

                if 'x_s0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'f_s' in d_outputs:
                d_out = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                for i in range(3):
                    d_out[i::3] = d_outputs['f_s'][i::self.struct_ndof]

                if 'u_s' in d_inputs:
                    d_in = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    # df_s/du_s^T * psi = - dL/du_s^T * psi
                    self.meld.applydLduSTrans(d_out,d_in)

                    for i in range(3):
                        d_inputs['u_s'][i::self.struct_ndof] -= np.array(d_in[i::3],dtype=float)

                if 'f_a' in d_inputs:
                    # df_s/df_a^T psi = - dL/df_a^T * psi = -dD/du_s * psi
                    prod = np.zeros(self.aero_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDduS(d_out,prod)
                    d_inputs['f_a'] -= np.array(prod,dtype=float)

                if 'x_a0' in d_inputs:
                    # df_s/dx_a0^T * psi = - psi^T * dL/dx_a0 in F2F terminology
                    prod = np.zeros(self.aero_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydLdxA0(d_out,prod)
                    d_inputs['x_a0'] -= np.array(prod,dtype=float)

                if 'x_s0' in d_inputs:
                    # df_s/dx_s0^T * psi = - psi^T * dL/dx_s0 in F2F terminology
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydLdxS0(d_out,prod)
                    d_inputs['x_s0'] -= np.array(prod,dtype=float)
