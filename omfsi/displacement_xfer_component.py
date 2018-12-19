from openmdao.api import ExplicitComponent

class FuntofemDisplacementTransfer(ExplicitComponent):
    """
    Component to perform displacement transfer using MELD
    """
    def initialize(self):
        self.options.declare('disp_xfer_setup', desc='function to instantiate MELD')

        self.options['distributed'] = True

        self.meld = None
        self.x_s0_old = None
        self.x_a0_old = None

    def setup(self):
        # get the transfer scheme object
        disp_xfer_setup = self.options['disp_xfer_setup']
        meld, aero_nnodes, struct_nnodes, struct_ndof = disp_xfer_setup(self.comm)
        self.meld = meld
        self.struct_ndof = struct_ndof

        # inputs
        self.add_input('x_s0', shape = struct_nnodes*3,           desc='initial structural node coordinates')
        self.add_input('x_a0', shape = aero_nnodes*3,             desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  shape = struct_nnodes*struct_ndof, desc='structural node displacements')

        # outputs
        self.add_output('u_a', shape = aero_nnodes*3,             desc='aerodynamic surface displacements')

        # partials
        #self.declare_partials('u_a',['x_s0','x_a0','u_s'])

    def compute(self, inputs, outputs):
        x_s0 = inputs['x_s0']
        x_a0 = inputs['x_a0']
        u_a  = outputs['u_a']

        u_s  = np.zeros(struct_nnodes*3)
        for i in range(3):
            u_s[i::3] = inputs['u_s'][i::self.struct_ndof]

        # give MELD the meshes and then form the connections
        if self.x_s0_old is None and self.x_a0_old is None:
            self.meld.setStructNodes(x_s0)
            self.meld.setAeroNodes(x_a0)
            self.meld.initialize()
            self.x_s0_old = x_s0.copy()
            self.x_a0_old = x_a0.copy()

        elif self._mesh_changed(x_s0,x_a0):
            self.meld.setStructNodes(x_s0)
            self.meld.setAeroNodes(x_a0)

        self.meld.transferDisps(u_s,u_a)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The MELD residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    # du_a/du_s^T * psi = - dD/du_s^T psi
                    prod = np.zeros(self.struct_nnodes*3)
                    self.meld.applydDduSTrans(d_outputs['u_a'],prod)
                    for i in range(3):
                        d_inputs['u_s'][i::self.struct_ndof] -= prod[i::3]

                # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                if 'x_a0' in d_inputs:
                    prod = np.zeros(d_inputs['x_a0'].size)
                    self.meld.applydDdxA0(d_outputs['u_a'],prod)
                    d_inputs['x_a0'] -= prod

                if 'x_s0' in d_inputs:
                    self.meld.applydDdxS0(d_outputs['u_a'],d_inputs['x_s0'])
                    d_inputs['x_s0'] -= prod

    def _mesh_changed(self,x_s0,x_a0):
        if not(np.allclose(x_s0,self.x_s0_old,rtol=1e-10,atol=1e-10) and
               np.allclose(x_a0,self.x_a0_old,rtol=1e-10,atol=1e-10)):
            self.x_s0_old = x_s0.copy()
            self.x_a0_old = x_a0.copy()
            return True
        else:
            return False
