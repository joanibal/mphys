# ======================================================================
#         Import modules
# ======================================================================


import os
import argparse
import numpy
from mpi4py import MPI
from baseclasses import *
# from tripan import TRIPAN
from pygeo_local import *
# from DVGeometry
# import pyBlock
from pyspline import *
from multipoint import *
from repostate import *
from pyoptsparse import Optimization, OPT

# ======================================================================
#         Input Information
# ======================================================================

import numpy as np
from pywarpustruct import *
from adflow import ADFLOW
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt



# ====================== Setup ======================

gridFile = './Meshes/rans_overset_coarse_old.cgns'
gridFile_cruise = './Meshes/cruiseGrid.cgns'
outputDirectory = '../../Research/Multi_Element_Opt/'

nGroup = 1
nProcPerGroup = 1
nFlowCases = 1

iteration = [0]




AEROSOLVER = ADFLOW

CFL = 5.6
MGCYCLE = 'sg'
MGSTART = 1
useNK = True

aeroOptions = {
    # Common Parameters
   'gridFile':gridFile,
   'outputDirectory':outputDirectory,
#    'restartfile':'fc_restart_vol_dep.cgns',

   
   # Physics Parameters
   'equationType':'rans',
   'smoother':'dadi',
   'useANKSolver': True,
   'nsubiterturb':7,
 
   'nsubiter':1,
   'resaveraging':'noresaveraging',
   'vis2':0.00,

    # Physics Parameters
    'equationType':'rans',
    # 'smoother':'rungakutta',
    'liftIndex':2,
    'CFL':CFL,
    'CFLCoarse':CFL,
    'MGCycle':MGCYCLE,
    'MGStartLevel':MGSTART,
    'nCyclesCoarse':250,
    'nCycles':9000,
    'monitorvariables':['resrho','cl','cd'],
    'volumevariables':['mach','resrho','blank'],
    'surfacevariables':['cp','vx', 'vy','vz', 'mach','blank'],
    'nearWallDist':0.0001,
    # 'nsubiterturb':3,
    'useNKSolver':useNK,
    'nkswitchtol':1.0e-5,
    'nrkreset': 50,

    # Convergence Parameters
    'L2Convergence':1e-14,
    # 'L2Convergence':1e-1,

    
    # Adjoint Parameters
    'adjointL2Convergence':1e-8,
    # 'adjointL2Convergence':5e-1,
    'ADPC':False,
    'adjointMaxIter': 500,
    'adjointSubspaceSize':150, 
    'ILUFill':2,
    'ASMOverlap[0]':1,
    'outerPreconIts':3,

    # Debugging parameters
    'debugzipper':False,

    'storerindlayer':False,
}
aeroOptions_cruise = {
   # Common Parameters
   'gridFile':gridFile_cruise,
   'outputDirectory':outputDirectory,
#    'restartfile':'fc_restart_vol_crusie.cgns',

   # Physics Parameters
   'equationType':'rans',
   'smoother':'dadi',


   #  'smoother':'runge kutta',
   'nsubiter':1,
   'liftIndex':3,
   'resaveraging':'noresaveraging',
   'vis2':0.00,

    # Physics Parameters
    'equationType':'rans',
    # 'smoother':'rungakutta',
    'liftIndex':2,
    'CFL':1.5    ,
    'CFLCoarse':3.0,
    'MGCycle':'sg',
    # 'MGStartLevel':2,
    # 'MGStartLevel':-1,
    'nCyclesCoarse':500,
    'nCycles':10000,
    'monitorvariables':['resrho','cl','cd', 'resturb'],
    'volumevariables':['mach','resrho','blank'],
    'surfacevariables':['cp','vx', 'vy','vz', 'mach','blank'],
    'nearWallDist':0.0001,
    # 'nsubiterturb':3,
    'useNKSolver':useNK,
    # 'nkswitchtol':2.750e-5,
    'nkswitchtol':1.0e-7,
    'rkreset':True,
    # 'nrkreset':50,
    'nsubiterturb':20,
   'useANKSolver': True,
    'ANKswitchtol': 1e-2,
   'ANKsecondordswitchtol' : 1e-6,

    # Convergence Parameters   
    'L2Convergence':1e-14
    ,
    'L2ConvergenceCoarse':1e-4,


    
    # Adjoint Parameters
    'adjointL2Convergence':1e-8,
    'ADPC':False,
    'adjointMaxIter': 500,
    'adjointSubspaceSize':150, 
    'ILUFill':2,
    'ASMOverlap[0]':1,
    'outerPreconIts':3,

    # Debugging parameters
    'debugzipper':False,

    'storerindlayer':False,

}
meshoptions = {
    'gridFile':gridFile,
    'symmetrySurfaces':['sym'],
    # 'symmetryPlanes':[[[1 ,0, 0],[0, 0, -1]],[[0,0, 1],[0.0, 0.0, 1.0]]],
    'warpType':'algebraic'
    }


meshoptions_cruise = {
    'gridFile':gridFile_cruise,
    'symmetrySurfaces':['sym'],
    # 'symmetryPlanes':[[[1 ,0, 0],[0, 0, -1]],[[0,0, 1],[0.0, 0.0, 1.0]]],
    'warpType':'algebraic'
    }


optOptions = {
           'Major feasibility tolerance':1e-4,
           'Major optimality tolerance':1e-4,

           # 'Iterations limit':,
           'Major step limit':1e-3,
           # 'Major step limit':1.0,
           # 'Function precision':1.0e-7,
           # 'Print file':os.path.join(iterDir, 'SNOPT_print.out'),
           # 'Summary file':os.path.join(iterDir, 'SNOPT_summary.out'),
           # 'Problem Type':'Minimize',
           # 'New superbasics limit':500,
           # 'Penalty parameter':1e3,
           }



# ======================================================================
#         Create multipoint communication object
# ======================================================================
MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet('cruise', nMembers=nGroup, memberSizes=nProcPerGroup)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

# Call  geometry setup
execfile('./setup_geometry_cruise.py')

execfile('./setup_geometry_deployed.py')

# ap = [AeroProblem(name='fc_deployed', mach=0.2, reynolds=9e6, reynoldsLength=1.0 , T=300, alpha=16, chordRef=1.00, areaRef=1.0, evalFuncs=[ 'cd']),\
#  AeroProblem(name='fc_cruise', mach=0.7, reynolds=1e9, reynoldsLength=1.0 , T=300, alpha=2.0, chordRef=1.00, areaRef=1.0, evalFuncs=[ 'cd'])]
# ap[0].addDV('alpha', value=16, lower=16, upper=16.0, scale=1.0) 
# ap[1].addDV('alpha', value=2.0, lower=2.0, upper=2.0, scale=1.0)
# ap = [ AeroProblem(name='fc_cruise', mach=0.7, reynolds=1e9, reynoldsLength=1.0 , T=300, alpha=2.0, chordRef=1.00, areaRef=1.0, evalFuncs=[ 'cd'])]


# Area_ref = 594720*.0254**2/2.0
# Span_ref = 2313.50*.0254
# Chord_ref =  275.8*.0254
# gamma = 1.4

# ref = Reference('Baseline Reference',Area_ref,Span_ref,, xref=3.968)

ap = [AeroProblem(name='fc_cruise', mach=0.65, alpha=1.0, altitude=37000*0.3048, chordRef=1.00, areaRef=1.0, evalFuncs=['cl', 'cd'])]
ap_deployed = [AeroProblem(name='fc_deployed', mach=0.2, reynolds=9e6, reynoldsLength=1.0 , T=300, alpha=16, chordRef=1.00, areaRef=1.0, evalFuncs=['cl'])]
ap[0].addDV('alpha', value=1.0, lower=-2.0, upper=2.0, scale=1.0e-1)
ap_deployed[0].addDV('alpha', value=16, lower=14, upper=20.0, scale=1.0e-1) 

# span = 1.0
# pos = numpy.array([0.5])*span
# CFDSolver.addSlices('z', pos,sliceType='absolute')

x_dv = {\
        'alpha_fc_cruise': numpy.array([0.5938791812665185]),\
        'alpha_fc_deployed': numpy.array([18.657948642520918]),\
        'shape': numpy.array([\
       -0.00396818,  0.00395309, -0.00396818,  0.00395309, -0.01010993,\
        0.01444647, -0.01010993,  0.01444647, -0.01028194, -0.01000274,\
       -0.01304514, -0.00735417, -0.01207038, -0.01587658, -0.01411869,\
       -0.01076799, -0.00736184, -0.0054498 , -0.00400901, -0.00344868,\
       -0.005494  , -0.00702922, -0.00988109, -0.00895572, -0.00749775,\
       -0.00548845,  0.0123001 ,  0.00794006, -0.0006147 , -0.00558138,\
       -0.00583797, -0.00608356, -0.00133957,  0.00783268,  0.01159536,\
        0.01253516,  0.01175349,  0.00988712,  0.00975464,  0.01164975,\
        0.01513668,  0.01512473,  0.01403973,  0.0112066 , -0.01028194,\
       -0.01000274, -0.01304514, -0.00735417, -0.01207038, -0.01587658,\
       -0.01411869, -0.01076799, -0.00736184, -0.0054498 , -0.00400901,\
       -0.00344868, -0.005494  , -0.00702922, -0.00988109, -0.00895572,\
       -0.00749775, -0.00548845,  0.0123001 ,  0.00794006, -0.0006147 ,\
       -0.00558138, -0.00583797, -0.00608356, -0.00133957,  0.00783268,\
        0.01159536,  0.01253516,  0.01175349,  0.00988712,  0.00975464,\
        0.01164975,  0.01513668,  0.01512473,  0.01403973,  0.0112066 ]),\
        'flap_twist': numpy.array([ 0.]),\
        'flap_move_x': numpy.array([ 0.]),\
        'flap_move_y': numpy.array([ 0.]),\
        'slat_twist': numpy.array([ 0.]),\
        'slat_move_x': numpy.array([ 0.]),\
        'slat_move_y': numpy.array([ 0.])   }
# note, need to add a single point so there is something to project to and call the update
# DVCon.writeTecplot('constraint_pts_check.dat')
#print(DVGeo.getVarNames())
DVGeo_cruise.addPointSet(np.array([[0,0,0]]), "foo")

DVGeo_cruise.update('foo')

DVGeo_cruise.writeTecplot('geo_test_1.dat')



DVGeo_cruise.setDesignVars(x_dv)
DVGeo_cruise.update('foo')
DVGeo_cruise.writeTecplot('geo_test_2.dat')
