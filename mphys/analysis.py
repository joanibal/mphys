""" this file holder the bases classes used by mphys"""
from openmdao.core.system import System
from openmdao.api import Group



# use parallele analysis for 
class Analysis(Group):
    def initialize(self):
        # super().initialize(self)
        self.solvers_init = False
        self.options.declare('share_solver_objects', default=True)

        self.objBuilders = []
        self.objTypeToBuilders = {}

    def setup(self):
        
        # loop over objbuilders
        for b in self.objBuilders:
            if b.obj is None:
                print(b)
                b.obj = b.build_obj(self.comm)

        # pass the obj to the builders of sub systems
        set_static_subs = set(self._subsystems_allprocs)
        set_subs = set(self._static_subsystems_allprocs)
        print( 'subs')
        for sub in set_static_subs | set_subs:

            if hasattr(sub, 'objBuilders'):
                for b in sub.objBuilders:
                    if b.obj_type in self.objTypeToBuilders:
                        b.obj = self.objTypeToBuilders[b.obj_type].obj

    def add_subsystem(self, *args, **kwargs):
        super().add_subsystem(*args, **kwargs)
        if hasattr(args[1], 'objBuilders'):
            for b in args[1].objBuilders:
                if not b.obj_type in self.objTypeToBuilders:
                    self.objTypeToBuilders[b.obj_type] = b
                    self.objBuilders.append(b)
