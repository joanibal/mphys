""" tests the analysis and the derivatives for the components wrapped by
builders in mphys"""





import openmdao.api as om
import numpy as np
from mpi4py import MPI

# try:
from parameterized import parameterized, parameterized_class
# except ImportError:
#     from openmdao.utils.assert_utils import SkipParameterized as parameterized
import unittest

from openmdao.utils.assert_utils import assert_near_equal

from mphys.mphys_meld import MELDThermal_heat_xfer_rate_xfer, MELDThermal_temp_xfer
# from mphys.mphys_rlt import RltBuilder

import mphys

xfer_options = {'isym': 1,
                'n': 10,
                'beta': 0.5}
rlt_options = {'transfergaussorder': 2}

# xfer_builder = MELD_builder(xfer_options, aero_builder, struct_builder)

# @parameterized_class(('name', 'xfer_builder_class','xfer_options' ), [
#    ('meld', MeldBuilder, meld_options),
# #    ('rlt', RltBuilder, rlt_options) # RLT can't take a dummy solver as input. It will be tested in regression tests
#    ])
class TestXferClasses(unittest.TestCase):
    def setUp(self):

        class FakeStructMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('x_s',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['x_s'] = self.nodes

        class FakeAeroMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('x_a',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['x_a'] = self.nodes

        class FakeCond(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)
                self.nodes = np.arange(12)

            def setup(self):
                self.add_input('q', shape=self.nodes.size)
                self.add_output('temp',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['temp'] = self.nodes

        class FakeConv(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_input('temp', shape=self.nodes.size)
                self.add_output('q',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['q'] = self.nodes

        np.random.seed(0)
        prob = om.Problem()

        heatflux_xfer_sys = MELDThermal_heat_xfer_rate_xfer(solver_options = xfer_options, check_partials=True)
        prob.model.add_subsystem('struct_mesh',FakeStructMesh())
        aero_mesh = prob.model.add_subsystem('aero_mesh',FakeAeroMesh())
        # prob.model.add_subsystem('struct_disps',FakeStructDisps())
        prob.model.add_subsystem('conv',FakeConv())
        prob.model.add_subsystem('heat_xfer_xfer', heatflux_xfer_sys)
        prob.model.add_subsystem('cond',FakeCond())

        prob.model.connect('conv.q', ['heat_xfer_xfer.heat_xfer_conv'])
        prob.model.connect('aero_mesh.x_a', ['heat_xfer_xfer.x_conv0'])
        prob.model.connect('struct_mesh.x_s', ['heat_xfer_xfer.x_cond0'])
        prob.model.connect('heat_xfer_xfer.heat_xfer_cond', ['cond.q'])


        prob.setup(force_alloc_complex=True)
        self.prob = prob
        om.n2(prob, show_browser=False, outfile='test_meld.html')

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        data = self.prob.check_partials()


        # there is an openmdao util to check partiacles, but we can't use it
        # because only SOME of the fwd derivatives are implemented
        for key, comp in data.items():
            for var, err  in comp.items():


                rel_err = err['rel error'] #,  'rel error']

                assert_near_equal(rel_err.reverse, 0.0, 5e-6)

                if var[1] == 'f_a' or var[1] == 'u_s':
                    assert_near_equal(rel_err.forward, 0.0, 5e-6)

class TestXferClasses(unittest.TestCase):
    def setUp(self):

        class FakeStructMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('x_s',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['x_s'] = self.nodes

        class FakeAeroMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('x_a',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['x_a'] = self.nodes

        class FakeCond(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)
                self.nodes = np.arange(12)

            def setup(self):
                self.add_input('q', shape=self.nodes.size)
                self.add_output('temp',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['temp'] = self.nodes

        class FakeConv(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_input('temp', shape=self.nodes.size)
                self.add_output('q',shape=self.nodes.size)

            def compute(self,inputs,outputs):
                outputs['q'] = self.nodes

        np.random.seed(0)
        prob = om.Problem()

        temp_xfer_sys = MELDThermal_temp_xfer(solver_options = xfer_options, check_partials=True)
        prob.model.add_subsystem('struct_mesh',FakeStructMesh())
        aero_mesh = prob.model.add_subsystem('aero_mesh',FakeAeroMesh())
        # prob.model.add_subsystem('struct_disps',FakeStructDisps())
        prob.model.add_subsystem('cond',FakeCond())
        prob.model.add_subsystem('temp_xfer', temp_xfer_sys)
        prob.model.add_subsystem('conv',FakeConv())

        prob.model.connect('cond.temp', ['temp_xfer.temp_cond'])
        prob.model.connect('aero_mesh.x_a', ['temp_xfer.x_conv0'])
        prob.model.connect('struct_mesh.x_s', ['temp_xfer.x_cond0'])
        prob.model.connect('temp_xfer.temp_conv', ['conv.temp'])


        prob.setup(force_alloc_complex=True)
        self.prob = prob
        om.n2(prob, show_browser=False, outfile='test_meld.html')

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        data = self.prob.check_partials()


        # # there is an openmdao util to check partiacles, but we can't use it
        # # because only SOME of the fwd derivatives are implemented
        # for key, comp in data.items():
        #     for var, err  in comp.items():


        #         rel_err = err['rel error'] #,  'rel error']

        #         assert_near_equal(rel_err.reverse, 0.0, 5e-6)

        #         if var[1] == 'f_a' or var[1] == 'u_s':
        #             assert_near_equal(rel_err.forward, 0.0, 5e-6)



if __name__ == '__main__':
    unittest.main()