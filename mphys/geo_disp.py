import numpy as np
import openmdao.api as om

class Geo_Disp(om.ExplicitComponent):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):

        self.add_input('x_a0',shape_by_connect = True, src_indices_by_connect = True, 
                                        desc='aerodynamic surface with geom changes')
        self.add_input('u_a', shape_by_connect = True, src_indices_by_connect = True,
                                             desc='aerodynamic surface displacements')

        self.add_output('x_a',determine_shape = True, desc='deformed aerodynamic surface')

    def determine_shape(self, var_meta_data):
        self.set_variable_shape('x_a', var_meta_data['x_a0'].shape)



    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_outputs['x_a'] += d_inputs['x_a0']
                if 'u_a' in d_inputs:
                    d_outputs['x_a'] += d_inputs['u_a']
        if mode == 'rev':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_inputs['x_a0'] += d_outputs['x_a']
                if 'u_a' in d_inputs:
                    d_inputs['u_a']  += d_outputs['x_a']