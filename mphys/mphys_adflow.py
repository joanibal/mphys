import numpy as np
import pprint
import copy

from collections import OrderedDict
from baseclasses import AeroProblem

from adflow import ADFLOW, ADFLOW_C
from idwarp import USMesh, USMesh_C

from openmdao.api import Group, ImplicitComponent, ExplicitComponent

from adflow.om_utils import get_dvs_and_cons

from .base_classes import ObjBuilder, SysBuilder
from .analysis import Analysis, SharedObjGroup


class AdflowObjBuilder(ObjBuilder):
    def __init__(self, options, func=None, complexify=False):
        super().__init__(options)
        self.func = func
        self.complexify = complexify

        self.obj_built = False

    def get_obj(self, comm):

        if not self.obj_built:
            # TODO the user should be able to choose the kind of mesh
            if self.complexify:
                mesh = USMesh_C(options=self.options["idwarp"], comm=comm)
                CFDSolver = ADFLOW_C(options=self.options["adflow"], comm=comm)
            else:
                mesh = USMesh(options=self.options["idwarp"], comm=comm)
                CFDSolver = ADFLOW(options=self.options["adflow"], comm=comm)
                
            if self.func:
                self.func(CFDSolver)
            
            CFDSolver.setMesh(mesh)
            self.solver = CFDSolver

            self.obj_built = True


        return self.solver

class AeroProblemMixIns:
    """
    a set of methods common the the components in the ADflow Group which use
    aeroproblems
    """

    def set_ap_design_vars(self, ap,  inputs):

        ap_vars, _ = get_dvs_and_cons(ap=ap)
        tmp = {}

        for (args, kwargs) in ap_vars:
            name = args[0]
            tmp[name] = inputs[name]

        ap.setDesignVars(tmp)

    def add_BCVars_to_ap(self,CFDSolver, ap, BCVar, famGroup):
        """
        adds any BCData from the CFDSolver to the aeroproblem

        Parameters
        ----------
        ap : AeroProblem
            The AeroProblem from MACH baseclasses hold the information about
            flow conditions which may be a design variables

        Returns
        ----------
        ap : AeroProblem
            The same problem, but with the BCVar added as a design variabl
        """
        bc_data = CFDSolver.getBCData()
        ap.setBCVar(BCVar, bc_data.getBCArraysFlatData(BCVar, familyGroup=famGroup), famGroup)
        ap.addDV(BCVar, familyGroup=famGroup, name=(BCVar + "_" + famGroup))


        return ap

    def add_ap_inputs(self, ap):
        """The design variables of the aero problem are set as inputs
        to the component

        Parameters
        ----------
        ap : AeroProblem
            The AeroProblem from MACH baseclasses hold the information about
            flow conditions which may be a design variables
        """
        # self.ap = ap

        ap_vars, _ = get_dvs_and_cons(ap=ap)
        irank = self.comm.rank

        if self.comm.rank == 0:
            print("adding ap var inputs")
        for (args, kwargs) in ap_vars:
            name = args[0]
            size = args[1]

            val = kwargs["value"]

            # the value of the ap variable may be scattered across the procs,
            # so we need to determine the src indices of the values we have
            s_list = self.comm.allgather(size)
            s1 = np.sum(s_list[:irank])
            s2 = np.sum(s_list[: irank + 1])

            self.add_input(name, val=val, src_indices=np.arange(s1, s2, dtype=int), units=kwargs["units"])
            # if self.comm.rank == 0:
            print(irank, name, size, s1, s2)

    def add_ap_outputs(self, ap, units=None):
        # this is the external function to set the ap to this component

        if units is None:
            units = {}

        for f_name in ap.evalFuncs:

            if self.comm.rank == 0:
                print("adding adflow func as output: {}".format(f_name))

            if f_name in units:
                self.add_output(f_name, shape=1, units=units[f_name])
            else:
                self.add_output(f_name, shape=1)
            # self.add_output(f_name, shape=1, units=units)



class AdflowMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates from an
    adflow object

    """

    def initialize(self):
        # self.options.declare('solver_options', default= {})

        self.options.declare("family_groups", default=["allWalls"])
        self.options.declare("obj_builders", default={AdflowObjBuilder: None}, recordable=False)
        self.options.declare("triangulated_mesh", default=False)
        self.options["distributed"] = True

    def setup(self):

        self.solver = self.options["obj_builders"][AdflowObjBuilder].get_obj(self.comm)

        for famGroup in self.options["family_groups"]:
            coords = self.solver.getSurfaceCoordinates(groupName=famGroup).flatten(order="C")
            self.add_output("X_%s" % (famGroup), val=coords, desc="surface points for %s" % (famGroup))

            if self.options["triangulated_mesh"]:
                tri_pts = self.solver.getTriangulatedMeshSurface(groupName=famGroup)

                self.add_output(f'X_{famGroup}_p0', val=tri_pts[0], desc="surface points for %s" % (famGroup))
                self.add_output(f'X_{famGroup}_v1', val=tri_pts[1], desc="surface points for %s" % (famGroup))
                self.add_output(f'X_{famGroup}_v2', val=tri_pts[2], desc="surface points for %s" % (famGroup))

class AdflowMapper(ExplicitComponent):
    """
    Component to get the partitioned surface mesh coordinates of different sub families

    """

    def initialize(self):
        # self.options.declare('solver_options', default= {})

        self.options.declare("input_family_group", default=["allWalls"])
        self.options.declare("output_family_groups", default=["allWalls"])

        self.options.declare("obj_builders", default={AdflowObjBuilder: None}, recordable=False)
        self.options["distributed"] = True

    def setup(self):

        self.solver = self.options["obj_builders"][AdflowObjBuilder].get_obj(self.comm)

        in_famGroup = self.options["input_family_group"]
        coords = self.solver.getSurfaceCoordinates(groupName=in_famGroup).flatten(order="C")
        self.add_input("X_%s" % (in_famGroup), val=coords, desc="surface points for %s" % (in_famGroup))

        for famGroup in self.options["output_family_groups"]:
            coords = self.solver.getSurfaceCoordinates(groupName=famGroup).flatten(order="C")
            self.add_output("X_%s" % (famGroup), val=coords, desc="surface points for %s" % (famGroup))

    def compute(self, inputs, outputs):

        in_famGroup = self.options["input_family_group"]
        in_vec = inputs["X_%s" % (in_famGroup)]
        in_vec = in_vec.reshape(in_vec.size // 3, 3)
        # loop over output families add do the mapping for each
        for out_famGroup in self.options["output_family_groups"]:

            out_vec = outputs["X_%s" % (out_famGroup)]
            out_vec = out_vec.reshape(out_vec.size // 3, 3)

            if self.solver.dtype == 'D' and not self.under_complex_step:
                in_vec = np.array(in_vec, dtype=self.solver.dtype)
                out_vec = np.array(out_vec, dtype=self.solver.dtype)

            out_vec = self.solver.mapVector(in_vec, in_famGroup, out_famGroup, out_vec)
            outputs["X_%s" % (out_famGroup)] = out_vec.flatten(order="C")


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        in_famGroup = self.options["input_family_group"]
        in_var_name = "X_%s" % (in_famGroup)

        if mode == "fwd":
            in_vec = d_inputs[in_var_name]
            in_vec = in_vec.reshape(in_vec.size // 3, 3)

            for out_famGroup in self.options["output_family_groups"]:
                if "X_%s" % (out_famGroup) in d_outputs:
                    out_vec = d_outputs["X_%s" % (out_famGroup)]
                    out_vec = out_vec.reshape(out_vec.size // 3, 3)
                    out_vec = self.solver.mapVector(in_vec, in_famGroup, out_famGroup, out_vec)

                    d_outputs["X_%s" % (out_famGroup)] += out_vec.flatten(order="C")

        elif mode == "rev":
            if in_var_name in d_inputs:
                for out_famGroup in self.options["output_family_groups"]:
                    if "X_%s" % (out_famGroup) in d_outputs:
                        out_vec = d_outputs["X_%s" % (out_famGroup)]
                        out_vec = out_vec.reshape(out_vec.size // 3, 3)

                        in_vec = np.zeros(d_inputs["X_%s" % (in_famGroup)].shape)
                        in_vec = in_vec.reshape(in_vec.size // 3, 3)
                        
                        # reverse the mapping!
                        self.solver.mapVector(out_vec, out_famGroup, in_famGroup, in_vec)

                        d_inputs[in_var_name] += in_vec.flatten(order="C")


class Geo_Disp(ExplicitComponent):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """

    def initialize(self):

        self.options["distributed"] = True

    def setup(self):

        # TODO set the val of the input x_a0
        self.add_input("x_a0", shape_by_conn=True, desc="aerodynamic surface with geom changes")
        self.add_input("u_a", shape_by_conn=True, desc="aerodynamic surface displacements")

        self.add_output("x_a", copy_shape="x_a0", desc="deformed aerodynamic surface")

    def compute(self, inputs, outputs):
        outputs["x_a"] = inputs["x_a0"] + inputs["u_a"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            if "x_a" in d_outputs:
                if "x_a0" in d_inputs:
                    d_outputs["x_a"] += d_inputs["x_a0"]
                if "u_a" in d_inputs:
                    d_outputs["x_a"] += d_inputs["u_a"]
        if mode == "rev":
            if "x_a" in d_outputs:
                if "x_a0" in d_inputs:
                    d_inputs["x_a0"] += d_outputs["x_a"]
                if "u_a" in d_inputs:
                    d_inputs["u_a"] += d_outputs["x_a"]


class AdflowWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.

    """

    def initialize(self):
        self.options.declare("obj_builders", default={AdflowObjBuilder: None}, recordable=False)

        self.options["distributed"] = True

    def setup(self):

        self.solver = self.options["obj_builders"][AdflowObjBuilder].get_obj(self.comm)

        # state inputs and outputs
        local_coords = self.solver.getSurfaceCoordinates()
        local_coord_size = local_coords.size
        local_volume_coord_size = self.solver.mesh.getSolverGrid().size

        n_list = self.comm.allgather(local_coord_size)
        irank = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[: irank + 1])

        self.add_input("x_a", src_indices=np.arange(n1, n2, dtype=int), shape=local_coord_size)

        self.add_output("x_g", shape=local_volume_coord_size)

    def compute(self, inputs, outputs):

        x_a = inputs["x_a"].reshape((-1, 3))
        self.solver.setSurfaceCoordinates(x_a)
        self.solver.updateGeometryInfo()
        outputs["x_g"] = self.solver.mesh.getSolverGrid().flatten(order="C")

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            if "x_g" in d_outputs:
                if "x_a" in d_inputs:
                    dxS = d_inputs["x_a"]
                    dxV = self.solver.mesh.warpDerivFwd(dxS)
                    d_outputs["x_g"] += dxV

        elif mode == "rev":
            if "x_g" in d_outputs:
                if "x_a" in d_inputs:
                    dxV = d_outputs["x_g"]
                    self.solver.mesh.warpDeriv(dxV)
                    dxS = self.solver.mesh.getdXs()
                    d_inputs["x_a"] += dxS.flatten()


class AdflowSolver(ImplicitComponent, AeroProblemMixIns):
    """
    OpenMDAO component that wraps the Adflow flow solver

    """

    def initialize(self):
        self.options.declare("obj_builders", default={AdflowObjBuilder: None}, recordable=False)
        self.options.declare("aero_problem")
        self.options.declare("BCDesVar", default={})

        self.options["distributed"] = True

    def setup(self):

        self.solver = self.options["obj_builders"][AdflowObjBuilder].get_obj(self.comm)

        # state inputs and outputs
        local_state_size = self.solver.getStateSize()
        local_coord_size = self.solver.mesh.getSolverGrid().size

        n_list = self.comm.allgather(local_coord_size)
        irank = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[: irank + 1])

        self.add_input("x_g", src_indices=np.arange(n1, n2, dtype=int), shape=local_coord_size)
        self.add_output("q", shape=local_state_size)

        self.ap = self.options["aero_problem"]
        for BCVar, famGroup in self.options["BCDesVar"].items():
            self.add_BCVars_to_ap(self.solver, self.ap, BCVar, famGroup)

        self.add_ap_inputs(self.ap)

    def apply_nonlinear(self, inputs, outputs, residuals):

        self.solver.setStates(outputs["q"])
        self.set_ap_design_vars(self.ap, inputs)

        ap = self.ap

        # Set the warped mesh
        self.solver.adflow.warping.setgrid(inputs["x_g"])

        # flow residuals
        residuals["q"] = self.solver.getResidual(ap)

    def solve_nonlinear(self, inputs, outputs):

        # Set the warped mesh
        self.solver.adflow.warping.setgrid(inputs["x_g"])
        self.set_ap_design_vars(self.ap, inputs)

        # reset the fail flags
        self.ap.solveFailed = False  # might need to clear this out?
        self.ap.fatalFail = False

        self.solver(self.ap, writeSolution=False)

        outputs["q"] = self.solver.getStates()

    def linearize(self, inputs, outputs, residuals):

        self.set_ap_design_vars(self.ap, inputs)
        self.solver.setStates(outputs["q"])

        # check if we changed APs, then we have to do a bunch of updates
        if self.ap != self.solver.curAP:
            # AP is changed, so we have to update the AP and
            # run a residual to make sure all intermediate vairables are up to date
            # we assume the AP has the last converged state information,
            # which is automatically set in the getResidual call
            self.solver.getResidual(self.ap)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        
        self.linearize(inputs, outputs, None)
        
        if mode == "fwd":
            if "q" in d_residuals:
                xDvDot = {}
                for var_name in d_inputs:
                    xDvDot[var_name] = d_inputs[var_name]
                if "x_g" in d_inputs:
                    xVDot = d_inputs["x_g"]
                else:
                    xVDot = None
                if "q" in d_outputs:
                    wDot = d_outputs["q"]
                else:
                    wDot = None

                dwdot = self.solver.computeJacobianVectorProductFwd(
                    xDvDot=xDvDot, xVDot=xVDot, wDot=wDot, residualDeriv=True
                )
                d_residuals["q"] += dwdot

        elif mode == "rev":
            if "q" in d_residuals:
                resBar = d_residuals["q"]

                wBar, xVBar, xDVBar = self.solver.computeJacobianVectorProductBwd(
                    resBar=resBar, wDeriv=True, xVDeriv=True, xDvDeriv=False, xDvDerivAero=True
                )
                
                if "q" in d_outputs:
                    d_outputs["q"] += wBar

                if "x_g" in d_inputs:
                    d_inputs["x_g"] += xVBar

                for dv_name, dv_bar in xDVBar.items():
                    if dv_name in d_inputs:
                        d_inputs[dv_name] += dv_bar.flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        
        # check if we changed APs, then we have to do a bunch of updates
        if self.ap != self.solver.curAP:
            # AP is changed, so we have to update the AP and
            # run a residual to make sure all intermediate vairables are up to date
            # we assume the AP has the last converged state information,
            # which is automatically set in the getResidual call
            self.solver.getResidual(self.ap)

        # the adjoint might not be set up regardless if we changed APs
        # this is because the first call with any AP will not have this set up, so we have to check
        # if we changed APs, then we also freed adjoint memory,
        # and then again we would need to setup adjoint again
        # finally, we generally want to avoid extra calls here
        # because this routine can be call multiple times back to back in a LBGS self.solver.
        if not self.solver.adjointSetup:
            self.solver._setupAdjoint()


        
        if mode == "fwd":
            d_outputs["q"] += self.solver.solveDirectForRHS(d_residuals["q"])
        elif mode == "rev":
            RHS = copy.deepcopy(d_outputs["q"])
            self.solver.adflow.adjointapi.solveadjoint(RHS, d_residuals["q"], True)

        return True, 0, 0


class AdflowFunctions(ExplicitComponent, AeroProblemMixIns):
    def initialize(self):
        self.options.declare("aero_problem")
        self.options.declare("BCDesVar", default={})


        self.options.declare("forces", default=False)
        self.options.declare("heatxfer", default=False)

        self.options.declare("obj_builders", default={AdflowObjBuilder: None}, recordable=False)
        self.options["distributed"] = True

        self.func_to_units = {
            "mdot": "kg/s",
            "mavgptot": "Pa",
            "mavgps": "Pa",
            "mavgttot": "degK",
            "mavgvx": "m/s",
            "mavgvy": "m/s",
            "mavgvz": "m/s",
            "drag": "N",
            "lift": "N",
            "dragpressure": "N",
            "dragviscous": "N",
            "dragmomentum": "N",
            "fx": "N",
            "fy": "N",
            "fz": "N",
            "forcexpressure": "N",
            "forceypressure": "N",
            "forcezpressure": "N",
            "forcexviscous": "N",
            "forceyviscous": "N",
            "forcezviscous": "N",
            "forcexmomentum": "N",
            "forceymomentum": "N",
            "forcezmomentum": "N",
            "flowpower": "W",
            "area": "m**2",
            "totheatflux": "W/m**2",
        }

    def setup(self):

        self.solver = self.options["obj_builders"][AdflowObjBuilder].get_obj(self.comm)

        local_state_size = self.solver.getStateSize()
        local_coord_size = self.solver.mesh.getSolverGrid().size
        s_list = self.comm.allgather(local_state_size)
        n_list = self.comm.allgather(local_coord_size)
        irank = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[: irank + 1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[: irank + 1])

        self.add_input("x_g", src_indices=np.arange(n1, n2, dtype=int), shape=local_coord_size)
        self.add_input("q", src_indices=np.arange(s1, s2, dtype=int), shape=local_state_size)

        self.ap = self.options["aero_problem"]
        for BCVar, famGroup in self.options["BCDesVar"].items():
            self.add_BCVars_to_ap(self.solver, self.ap, BCVar, famGroup)


        self.add_ap_inputs(self.ap)
        self.add_ap_outputs(self.ap, self.func_to_units)

        if self.options["forces"]:
            local_surface_coord_size = self.solver._getSurfaceSize(self.solver.allWallsGroup)
            self.add_output("f_a", shape=local_surface_coord_size)
        if self.options["heatxfer"]:
            local_nodes, nCells = self.solver._getSurfaceSize(self.solver.allIsothermalWallsGroup)

            self.add_output("heatflux", val=np.ones(local_nodes) * 0, shape=local_nodes, units="W/m**2")


    def _get_func_name(self, name):
        return "%s_%s" % (self.ap.name, name.lower())

    def compute(self, inputs, outputs):
        self.set_ap_design_vars(self.ap, inputs)

        # Set the warped mesh
        self.solver.adflow.warping.setgrid(inputs["x_g"])

        self.solver.setStates(inputs["q"])

        funcs = {}
        eval_funcs = [f_name for f_name in self.ap.evalFuncs]

        self.solver.evalFunctions(self.ap, funcs, eval_funcs)

        for name in self.ap.evalFuncs:
            f_name = self._get_func_name(name)
            if f_name in funcs:
                outputs[name.lower()] = funcs[f_name]

        if self.options["forces"]:
            outputs["f_a"] = self.solver.getForces().flatten(order="C")

        if self.options["heatxfer"]:
            outputs["heatflux"] = self.solver.getHeatXferRates().flatten(order="C")


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        self.set_ap_design_vars(self.ap, inputs)


        # check if we changed APs, then we have to do a bunch of updates
        if self.ap != self.solver.curAP:
            # AP is changed, so we have to update the AP and
            # run a residual to make sure all intermediate vairables are up to date
            # we assume the AP has the last converged state information,
            # which is automatically set in the getResidual call
            self.solver.getResidual(self.ap)


        
        if mode == "fwd":
            xDvDot = {}
            for key in ap.DVs:
                if key in d_inputs:
                    mach_name = key.split("_")[0]
                    xDvDot[mach_name] = d_inputs[key]

            if "q" in d_inputs:
                wDot = d_inputs["q"]
            else:
                wDot = None

            if "x_g" in d_inputs:
                xVDot = d_inputs["x_g"]
            else:
                xVDot = None

            funcsdot, fdot, hfdot = self.solver.computeJacobianVectorProductFwd(
                xDvDot=xDvDot, xVDot=xVDot, wDot=wDot, funcDeriv=True, fDeriv=True, hfDeriv=True
            )

            if "f_a" in d_outputs:
                d_outputs["f_a"] += fdot.flatten()

            if "heatflux" in d_outputs:
                hftmp = np.zeros((hfdot.size, 3))
                hftmp[:, 0] = hfdot

                hfdot = self.solver.mapVector(hfdot, self.solver.allWallsGroup, self.solver.allIsothermalWallsGroup)
                hfdot = hfdot[:,0]
                d_outputs["heatflux"] += hfdot.flatten()

            for name in funcsdot:
                func_name = name.lower()
                if name in d_outputs:
                    d_outputs[name] += funcsdot[func_name]

        elif mode == "rev":
            funcsBar = {}

            for name in self.ap.evalFuncs:
                func_name = name.lower()

                # we have to check for 0 here, so we don't include any unnecessary variables in funcsBar
                # becasue it causes Adflow to do extra work internally even if you give it extra variables, even if the seed is 0
                if func_name in d_outputs and d_outputs[func_name] != 0.0:
                    funcsBar[func_name] = d_outputs[func_name][0]/self.comm.size # tmp fix while om is changing distibuted vars

            if "f_a" in d_outputs:
                fBar = d_outputs["f_a"]
            else:
                fBar = None

            if "heatflux" in d_outputs:
                hf = d_outputs["heatflux"]

                # make the vector into the look like a coordinate vector
                hfBar = np.zeros((hf.size, 3))
                hfBar[:, 0] = hf

                hfBar = self.solver.mapVector(hfBar, self.solver.allIsothermalWallsGroup, self.solver.allWallsGroup)
                hfBar = hfBar[:, 0]
            else:
                hfBar = None


            wBar, xVBar, xDVBar = self.solver.computeJacobianVectorProductBwd(
                funcsBar=funcsBar, fBar=fBar, hfBar=hfBar, wDeriv=True, xVDeriv=True, xDvDeriv=False, xDvDerivAero=True
            )

            if "q" in d_inputs:
                d_inputs["q"] += wBar
            if "x_g" in d_inputs:
                d_inputs["x_g"] += xVBar

            for dv_name, dv_bar in xDVBar.items():
                if dv_name in d_inputs:
                    d_inputs[dv_name] += dv_bar.flatten()



class ADflowWriteSolution(ExplicitComponent, AeroProblemMixIns):
    """
    This componeent is used only for outputing a solution file for postprocessing
    """
    def initialize(self):
        self.options.declare("obj_builders", default={AdflowObjBuilder: None}, recordable=False)
        self.options["distributed"] = True


    def setup(self):

        self.solver = self.options["obj_builders"][AdflowObjBuilder].get_obj(self.comm)

        local_state_size = self.solver.getStateSize()
        local_coord_size = self.solver.mesh.getSolverGrid().size
        s_list = self.comm.allgather(local_state_size)
        n_list = self.comm.allgather(local_coord_size)
        irank = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[: irank + 1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[: irank + 1])

        self.add_input("x_g", src_indices=np.arange(n1, n2, dtype=int), shape=local_coord_size)
        self.add_input("q", src_indices=np.arange(s1, s2, dtype=int), shape=local_state_size)
        
        self.counter = 0 


    def compute(self, inputs, outputs):

        # Set the warped mesh
        self.solver.adflow.warping.setgrid(inputs["x_g"])
        self.solver.setStates(inputs["q"])

        self.solver.writeSolution(number=self.counter)
        self.counter += 1 
        
class AdflowGroup(SharedObjGroup):
    def initialize(self):
        super().initialize()
        self.options.declare("aero_problem")
        self.options.declare("group_options")
        self.options.declare("forces", default=False)
        self.options.declare("heatxfer", default=False)
        self.options.declare("BCDesVar", default={})
        self.options["share_all_builders"] = False
        self.options.declare("shared_obj_builders", default={AdflowObjBuilder: None}, recordable=False)

        # default values which are updated later
        self.group_options = {
            "mesh": True,
            "geo_disp": False,
            "deformer": True,
            "solver": True,
            "funcs": True,
            # "forces": False,
            # "heatxfer": False,
        }
        self.group_components = OrderedDict(
            {
                "mesh": AdflowMesh,
                "geo_disp": Geo_Disp,
                "deformer": AdflowWarper,
                "solver": AdflowSolver,
                "funcs": AdflowFunctions,
                # "forces": AdflowForces,
                # "heatxfer": AdflowHeatTransfer,
            }
        )

        # self.solver_objects = {'Adflow':None}

        self.solvers_init = False

    def setup(self):
        # issue conditional connections
        self.group_options.update(self.options["group_options"])

        if self.group_options["mesh"] and self.group_options["deformer"] and not self.group_options["geo_disp"]:
            # import ipdb; ipdb.set_trace()
            self.connect("Xsurf_allWalls", "x_a")

        # if you wanted to check that the user gave a valid combination of components (solver, mesh, ect)
        # you could do that here, but they will be shown on the n2

        print("=========")
        for comp_name, comp in self.group_components.items():
            if self.group_options[comp_name]:
                print(comp_name)
                if comp_name in ["mesh", "geo_disp", "deformer"]:
                    comp = self.group_components[comp_name]()
                elif comp_name == "solver":
                    comp = self.group_components[comp_name](aero_problem=self.options["aero_problem"],
                                                            BCDesVar=self.options["BCDesVar"])
                elif comp_name == "funcs":
                    comp = self.group_components[comp_name](aero_problem=self.options["aero_problem"],
                                                            BCDesVar=self.options["BCDesVar"],
                                                            forces=self.options['forces'],
                                                            heatxfer=self.options["heatxfer"])

                self.add_subsystem(comp_name, comp, promotes=["*"])
                #  we can connect things implicitly through promotes
                #  because we already know the inputs and outputs of each
                #  components

        super().setup()
