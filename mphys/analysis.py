""" this file holder the bases classes used by mphys"""
from openmdao.core.system import System
from openmdao.api import Group


class SharedObjGroup(Group):
    def initialize(self):

        self.options.declare("share_all_builders", default=True)
        # the default behavoir is to pass any ObjBuilders that were pass to the group to the subsystems

        self.options.declare("shared_obj_builders", default={}, recordable=False)
        # if the share_all_builders is False instead it will only pass along obj builders that match the types of the dictionary
        # >>self.options.declare('shared_obj_builders', default={AeroObjBuilder: None})
        # >>self.options.declare('shared_obj_builders', default={AdflowObjBuilder: None})
        # builders that are None do not get placed into subsystems

    def setup(self):

        subinfo = {}
        subinfo.update(self._subsystems_allprocs)
        subinfo.update(self._static_subsystems_allprocs)

        # loop over the subsystems and pass builder objects as needed
        for s in subinfo.values():
            subsys = s.system

            if isinstance(subsys, SharedObjGroup):
                # if the subsys is also a shared object group
                if subsys.options["share_all_builders"]:
                    subsys.options["shared_obj_builders"].update(self.options["shared_obj_builders"])
                else:
                    for bldr_type, bldr_inst in self.options["shared_obj_builders"].items():
                        for subsys_bldr_type, subsys_bldr_inst in subsys.options["shared_obj_builders"].items():
                            if issubclass(subsys_bldr_type, bldr_type) and bldr_inst is not None:
                                # pass the builder to the subsystem
                                subsys.options["shared_obj_builders"][subsys_bldr_type] = bldr_inst

            else:  # the subsys is a component
                if "obj_builders" in subsys.options:

                    print("adding builder", self.options["shared_obj_builders"], "to ", subsys.name)

                    for bldr_type, bldr_inst in self.options["shared_obj_builders"].items():
                        for subsys_bldr_type, subsys_bldr_inst in subsys.options["obj_builders"].items():
                            # print(subsys_bldr_type, bldr_type)

                            if issubclass(subsys_bldr_type, bldr_type) and bldr_inst is not None:
                                # pass the builder to the subsystem
                                print("passing", bldr_inst)
                                subsys.options["obj_builders"][subsys_bldr_type] = bldr_inst


class Analysis(Group):
    def initialize(self):
        # super().initialize(self)
        self.solvers_init = False
        self.options.declare("share_solver_objects", default=True)

        self.objBuilders = []
        self.objTypeToBuilders = {}

    def setup(self):

        # loop over objbuilders
        for b in self.objBuilders:
            if b.obj is None:
                print(b)
                b.obj = b.build_obj(self.comm)

        # pass the obj to the builders of sub systems
        # set_static_subs = self._subsystems_allprocs)
        # set_subs = set(self._static_subsystems_allprocs)
        subinfo = {}
        subinfo.update(self._subsystems_allprocs)
        subinfo.update(self._static_subsystems_allprocs)
        # import ipdb; ipdb.set_trace()
        for s in subinfo.values():
            subsys = s.system
            if hasattr(subsys, "objBuilders"):
                for b in subsys.objBuilders:
                    if b.obj_type in self.objTypeToBuilders:
                        b.obj = self.objTypeToBuilders[b.obj_type].obj

    def add_subsystem(self, *args, **kwargs):
        super().add_subsystem(*args, **kwargs)
        if hasattr(args[1], "objBuilders"):
            for b in args[1].objBuilders:
                if b.obj_type not in self.objTypeToBuilders:
                    self.objTypeToBuilders[b.obj_type] = b
                    self.objBuilders.append(b)
