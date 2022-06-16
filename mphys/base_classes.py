""" this file holder the bases classes used by mphys"""

from abc import ABCMeta, abstractmethod

from openmdao.core.system import System
from openmdao.api import Group
from itertools import product, chain
from pprint import pprint


class SolverObjectBasedSystem(metaclass=ABCMeta):
    """
    the base class for analysis in MPhys that require a solver object.
    """

    # the following mush be set in the initialize method
    # def initialize(self):

    # initialize the solver objects to None
    # self.solver_objects = { 'MeshName':None,
    #                     'SolverName':None,
    #                     'EctName':None}

    # set the init flag to false
    # self.solvers_init = False

    @abstractmethod
    def init_solver_objects(self, comm):
        """creates solver/transfer scheme/etc using provided comm which
        entails the allocation of memory for any computation"""

        mesh = mymodule.MeshObj(self.options["mesh_file"])
        solver = mymodule.SolverObj(self.options["solver_options"])

        self.solver_objects = {"MeshName": mesh, "SolverName": solver, "EctName": ect}

        # set the init flag to true!
        self.solvers_init = True

    def get_solver_objects(self):
        self.check_init_solvers()
        return self.solver_objects

    def set_solver_objects(self, solver_objects):
        self.solver_objects.update(solver_objects)

        # if all of the dictionary values for the solver_objects dict are not none
        if all(self.solver_objects.values()):
            self.solvers_init = True

    def check_init_solvers(self):
        if not self.solvers_init:
            import ipdb

            ipdb.set_trace()
            raise RuntimeError("Solver used before it was initialized or set")


class ObjBuilder:
    def __init__(self, options):
        self.options = options

    # @property
    # def obj(self):
    #     # if self.solvers_obj is None:
    #     #     print('Solver used before it was initialized or set')
    #     #     import ipdb; ipdb.set_trace()
    #     #     raise RuntimeError('Solver used before it was initialized or set')

    #     return self._obj

    # @obj.setter
    # def obj(self, obj):
    #     # check that the obj is the right type
    #     if isinstance(obj, self.obj_type):
    #         self._obj = obj
    #     else:

    #         print('type of obj supplied does not match type expected')
    #         import ipdb; ipdb.set_trace()
    #         raise RuntimeError('Solver used before it was initialized or set')

    # def  build_obj(self, comm):
    #     """this must be added for all derived types"""

    #     raise NotImplementedError


class SysBuilder(object):
    def __init__(self, mesh_sys=None, sys=None, funcs_sys=None, options=None):
        self._mesh = mesh_sys
        self._sys = sys
        self._funcs = funcs_sys

        # options is a dictionary of options for the mesh, system, and func system
        # options = {'mesh': mesh_options, 'sys': sys_options, 'funcs':func options}
        self.options = options

    @property
    def mesh(self):
        # print(**self.options['mesh'])
        return self._mesh()

    @property
    def sys(self):
        return self._sys(**self.options["sys"])

    @property
    def funcs_sys(self):
        return self._funcs_sys(**self.options["funcs_sys"])


if __name__ == "__main__":
    builder = SysBuilder(mesh_sys=Group, sys=Group, funcs_sys=Group)

    import ipdb

    ipdb.set_trace()
