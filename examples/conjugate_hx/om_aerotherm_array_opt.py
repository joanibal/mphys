
from __future__ import division, print_function
from mpi4py import MPI
from openmdao.api import ExplicitComponent, IndepVarComp
import numpy as np

from openmdao.api import Problem
import openmdao.api as om
from openmdao.api import NonlinearRunOnce, LinearRunOnce


from OMFSI_myfork.adflow_component import *

from conduction_models import ConductionNodal

from multipoint import redirectIO
# My Modules
from pprint import pprint

# # import matplotlib.pyplot as plt
from helperFuncOpt import get_output_dirs

from options import getAeroOptions, opt_options, ffd_options
from setupGeometry import getDVGeo
from aeroProblemData import ap_runup, ap_cruise
from opt_dvs import opt_dvs_array_dict

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default=os.path.join(os.environ["DATAFOLDER"], 'nacelle_opt_array'))
parser.add_argument('--hot_start',  help='history file to restart with', type=str, default=None)
parser.add_argument('--redirect',  help='redirect output to txt file', action='store_true',)
parser.add_argument('--restart',   help='restart CFD from file', action='store_true',)
parser.add_argument('--tag',  help='additional info to add to output file name', type=str, default='')

parser.add_argument('--debug',   help='set debugging options', action='store_true')
parser.add_argument('--aero_only',   help='only do aero only opt', action='store_true')
parser.add_argument('--uncoupled',   help='only do aero only opt', action='store_true')
parser.add_argument('--cold_start',  help='history file to restart with',  action='store_true')


parser.add_argument('--step_size', help='step size', type=float, default=5e-3)
parser.add_argument('--min_heatflux', help='lower bound on heatflux (negated in func)', type=float, default=0)
parser.add_argument('--mesh_level', help='step size', type=str, default='L4')
parser.add_argument('--mode', help='step size', type=str, default='opt')
parser.add_argument('--scale_lower', help='lower bound on heatflux (negated in func)', type=float, default=1)


args = parser.parse_args()

# for using the cmd line tool 
# class Args(object):
#     def __init__(self):
#         self.output = os.path.join(os.environ["DATAFOLDER"], 'nacelle_opt_array')
#         self.debug = False
#         self.aero_only = False
#         self.step_size = 1e-1
#         self.min_heatflux = 0
#         self.mesh_level = 'L4'
#         self.mode = 'opt'
#         self.cold_start = False
#         self.uncoupled = False
#         self.tag = ''
#         self.restart = False
#         self.redirect = False

# args = Args()

# ap_cruise.setBCVar('Temperature', 276, 'isothermalwall')
# ap_runup.addDV('Temperature', family='isothermalwall', name='wall_temp')
# if args.mesh_level == 'L2':
#     shape = (4,9*9)
# elif args.mesh_level == 'L3':
#     shape = (4,5*5)
# elif args.mesh_level == 'L4':
#     shape = (4,3*3)
# temps = np.ones(shape)*399
# heatfluxes = np.ones(shape)*-501

                  


output_dir, cruise_dir, runup_dir  = get_output_dirs(args)
aero_options_runup  = getAeroOptions('./meshes/array_temp/nacelle_' + args.mesh_level + '.cgns', runup_dir, restart=args.restart, debug=args.debug)
aero_options_cruise = getAeroOptions('./meshes/nacelle_' + args.mesh_level + '.cgns', cruise_dir, restart=args.restart, debug=args.debug)


mesh_options = {
    'gridFile':aero_options_runup['gridFile'],
}
# if MPI.COMM_WORLD.rank == 0: # Only global root proc makes
#     cruise_dir = args.outputss
#     os.system('mkdir -p %s'%(aero_options_runup['output']))


if MPI.COMM_WORLD.rank == 0:
    
    pprint(args)
    if args.redirect:
        logFile = open(output_dir+ '/log.txt', 'w+b')
        print(output_dir)

        redirectIO(logFile)
        print(logFile.name + ' created sucessfully')
        pprint(args)
    
MPI.COMM_WORLD.barrier()

# this is becuase Dr jacobs does this
class AdflowSimpleAssembler(object):
    def __init__(self,ap,options):
        self.options = options
        self.solver = None
        self.ap = ap
    def get_solver(self,comm):
        if self.solver is None:
            print(self.ap.name)
            self.solver = ADFLOW(options=self.options)
            self.mesh = USMesh(comm=comm,options=mesh_options)
            self.solver.setMesh(self.mesh)
            self.solver.setDVGeo(getDVGeo(ffd_options))
            self.solver.setAeroProblem(self.ap)
            self.solver.name += self.ap.name
            self.solver.addSlices('y', 0.0, sliceType='absolute')
            self.solver.addSlices('z', 0.0, sliceType='absolute')
        return self.solver

# this is becuase Dr jacobs does this
class AdflowSimpleAssemblerArray(object):
    def __init__(self,ap,options):
        self.options = options
        self.solver = None
        self.ap = ap
    def get_solver(self,comm):
        if self.solver is None:
            print(self.ap.name)

            self.solver = ADFLOW(options=self.options)
            temps = self.solver.getWallTemperature()

            nNodes_motor = comm.allreduce(temps.size, op=MPI.SUM)

            if temps.size > 0 :

                nBlks = int(float(temps.size)/nNodes_motor*4)

                temps = np.reshape(temps, (nBlks, temps.size//(nBlks)))

                print(comm.rank, temps.shape, nBlks)
            # quit()
            self.ap.setBCVar('Temperature', temps, 'isothermalwall')
            self.ap.addDV('Temperature', family='isothermalwall', name='wall_temp')
            self.mesh = USMesh(comm=comm,options=mesh_options)
            self.solver.setMesh(self.mesh)
            self.solver.setDVGeo(getDVGeo(ffd_options))
            self.solver.setAeroProblem(self.ap)
            self.solver.name += self.ap.name
            self.solver.addSlices('y', 0.0, sliceType='absolute')
            self.solver.addSlices('z', 0.0, sliceType='absolute')

        return self.solver


 

assembler = AdflowSimpleAssemblerArray(ap_runup, aero_options_runup)
assembler_cruise = AdflowSimpleAssembler(ap_cruise, aero_options_cruise)
# Adflow components set up
geo_comp    = AdflowGeo(ap=ap_runup, get_solver=assembler.get_solver)
warp_comp   = AdflowWarper(ap=ap_runup,  get_solver=assembler.get_solver)
solver_comp_runup = AdflowSolver(ap=ap_runup,get_solver=assembler.get_solver)
heat_comp_runup   = AdflowHeatTransfer(ap=ap_runup,get_solver=assembler.get_solver)
conduc_comp = ConductionNodal(get_solver=assembler.get_solver)
func_comp_runup   = AdflowFunctions(ap=ap_runup,get_solver=assembler.get_solver)


warp_comp_cruise   = AdflowWarper(ap=ap_cruise,  get_solver=assembler_cruise.get_solver)
solver_comp_cruise = AdflowSolver(ap=ap_cruise,get_solver=assembler_cruise.get_solver)
func_comp_cruise   = AdflowFunctions(ap=ap_cruise,get_solver=assembler_cruise.get_solver)
prob = Problem()
model = prob.model

dvs = model.add_subsystem('dvs', om.IndepVarComp())
if args.cold_start:
    dv_0 = opt_dvs_array_dict[args.aero_only][args.uncoupled][args.mesh_level]
else: 
    dv_0 = np.ones(15)*1.10

dvs.add_output('scale_sections',dv_0)

model.add_subsystem('geo',geo_comp)
model.add_subsystem('warp',warp_comp)


if not args.aero_only:
    cycle = model.add_subsystem('cycle', Group())
    cycle.add_subsystem('solver_runup',solver_comp_runup)
    cycle.add_subsystem('heat_runup',heat_comp_runup)
    # cycle.connect('solver_runup.q',['heat_runup.q'])

    if not args.uncoupled:
        cycle.add_subsystem('conduc', conduc_comp)

        cycle.nonlinear_solver = om.NonlinearBlockGS()
        cycle.nonlinear_solver.options['maxiter'] = 15
        cycle.nonlinear_solver.options['atol'] = 1e-8
        cycle.nonlinear_solver.options['rtol'] = 1e-8
        cycle.nonlinear_solver.options['iprint'] = 2
        # cycle.nonlinear_solver.options['use_aitken'] = True

        if args.debug:
            cycle.nonlinear_solver.options['maxiter'] = 2

model.add_subsystem('func_runup',func_comp_runup)

model.add_subsystem('warp_cruise',warp_comp_cruise)
model.add_subsystem('solver_cruise',solver_comp_cruise)
model.add_subsystem('funcs_cruise',func_comp_cruise)


if not args.aero_only:
    if args.uncoupled:
        model.connect('geo.x_a0_mesh',['warp.x_a', 'warp_cruise.x_a'])
        # model.connect('dvs.wall_temp_runup',  ['solver_runup.wall_temp', 'heat_runup.wall_temp'])
    else:
        model.connect('cycle.solver_runup.q',['cycle.heat_runup.q', 'func_runup.q'])

        cycle.connect('heat_runup.heatflux', ['conduc.heatflux'])
        cycle.connect('conduc.T_surf', ['solver_runup.wall_temp', 'heat_runup.wall_temp'])

        model.connect('geo.x_a0_mesh',['warp.x_a',  'warp_cruise.x_a', 'cycle.conduc.x_a'])

    model.connect('warp.x_g',['cycle.solver_runup.x_g','cycle.heat_runup.x_g', 'func_runup.x_g'])
    
else: 
    model.connect('geo.x_a0_mesh',['warp.x_a',  'warp_cruise.x_a'])
   
model.connect('solver_cruise.q',['funcs_cruise.q'])
model.connect('warp_cruise.x_g',['solver_cruise.x_g','funcs_cruise.x_g'])



model.connect('dvs.scale_sections',['geo.scale_sections'])


prob.model.add_design_var('dvs.scale_sections',
  lower=np.array([args.scale_lower, args.scale_lower, args.scale_lower, args.scale_lower, args.scale_lower,\
                  1.0, 1.0, 1.0, args.scale_lower, args.scale_lower,\
                  args.scale_lower, args.scale_lower, args.scale_lower, args.scale_lower, args.scale_lower]), 
  upper=np.ones(15)*3)
prob.model.add_objective('funcs_cruise.cd', scaler=1e1)

if not args.aero_only:
    # prob.model.add_constraint('cycle.conduc.T_surf', lower=0, scaler=1e-2)
    prob.model.add_constraint('func_runup.heatflux', lower=-1*args.min_heatflux*1.5,  upper=-1*args.min_heatflux, scaler=1e2)




prob.set_solver_print(level=2)

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"

prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-3
prob.driver.opt_settings['Verify level'] = 0
# prob.driver.opt_settings['Hessian updates'] =20
prob.driver.opt_settings['Function precision'] = 1e-7

prob.driver.opt_settings['Print file'] = output_dir + '/print_snopt.out'
prob.driver.opt_settings['Summary file'] = output_dir + '/summary_snopt.out'
prob.driver.opt_settings['Major step limit'] = args.step_size

recorder = om.SqliteRecorder(output_dir + '/hist_opt.db')

# add the recorder to the driver so driver iterations will be recorded
# cycle.add_recorder(recorder, True)
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_responses'] = True
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_constraints'] = True


prob.setup()

###############################
if __name__ == "__main__":
    if args.mode == 'opt':
        prob.run_driver()
        prob.record_iteration('final')

    elif args.mode == 'runonce':
        model.nonlinear_solver = NonlinearRunOnce()
        model.linear_solver = LinearRunOnce()
        prob.run_model()

    elif args.mode == 'derivCheck':
        # prob.check_partials(includes=['cycle.conduc'])
        # p = om.Problem()
        # p.model.add_subsystem('conduc',conduc_comp)
        # p.setup()
        prob.check_partials(includes=['heat_runup'])

def main():
    prob.run_driver()

def deriv():
    p = om.Problem()
    p.model.add_subsystem('conduc',conduc_comp)
    p.setup()
    p.check_partials()

