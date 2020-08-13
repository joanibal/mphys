#!/usr/bin/env python
from __future__ import print_function, division
from mpi4py import MPI
import numpy as np
#import matplotlib.pyplot as plt
import sys
# Import SU2
import pysu2
from tacs import TACS, elements, constitutive

from adflow import ADFLOW
from baseclasses import AeroProblem
import os
import sys

# baseDir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(baseDir,'../../../../'))


from funtofem import TransferScheme

import matplotlib.pyplot as plt

# Options
prior_steps = 0                 # if restarting set to number of steps in the restart file
nsteps = 50                    # number of time steps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

thermal_coupling = True
verbose = False
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

aeroOptions = {
    # 'printTiming': False,

    # Common Parameters
    'gridFile': 'naca0012_hot_rans.cgns',
    'outputDirectory': './',
    # 'discretization': 'upwind',

    # 'oversetupdatemode': 'full',
    'volumevariables': ['temp'],
    'surfacevariables': ['cf', 'vx', 'vy', 'vz', 'temp', 'heattransfercoef', 'heatflux'],
    'monitorVariables':	['resturb', 'yplus', 'heatflux'],
    # Physics Parameters
    # 'equationType': 'laminar NS',
    'equationType': 'rans',
    # 'vis2':0.0,
    'liftIndex': 2,
    'CFL': 1.0,
    # 'smoother': 'DADI',
    # 'smoother': 'runge',

    'useANKSolver': True,
    'ANKswitchtol': 10e0,
    # 'ankcfllimit': 5e6,
    'anksecondordswitchtol': 5e-3,
    'ankcoupledswitchtol': 1e-7,
    # NK parameters
    'useNKSolver': True,
    'nkswitchtol': 1e-5,
    
    'rkreset': False,
    'nrkreset': 40,
    'MGCycle': 'sg',
    # 'MGStart': -1,
    # Convergence Parameters
    'L2Convergence': 1e-12,
    'nCycles': 100,
    'nCyclesCoarse': 250,
    'ankcfllimit': 5e3,
    'nsubiterturb': 5,
    'ankphysicallstolturb': 0.99,
    'anknsubiterturb': 5,
    # 'ankuseturbdadi': False,
    'ankturbkspdebug': True,

    'storerindlayer': True,
    # Turbulence model
    'eddyvisinfratio': .210438,
    'useft2SA': False,
    'turbulenceproduction': 'vorticity',
    'useblockettes': False,

}

# atmospheric conditions
temp_air = 273  # kelvin
Pr = 0.72
mu = 1.81e-5  # kg/(m * s)

u_inf = 10  # m/s\
p_inf = 101e3


rho_inf = p_inf/(287*temp_air)
CFDSolver = ADFLOW(options=aeroOptions)


ap = AeroProblem(name='fc_conv', V=u_inf, T=temp_air,
                rho=1.225, areaRef=1.0, chordRef=1.0, alpha=10.0, beta=0,  evalFuncs=['cl', 'cd'])


BCVar = 'Temperature'
nAeroNodes = []
XAero = []
hot_groups = ['heated_wall_bottom' , 'heated_wall_top']
for group in hot_groups:
    X = CFDSolver.getSurfacePoints(groupName=group)
    BCData = CFDSolver.getBCData(groupNames=[group])

    # import ipdb; ipdb.set_trace()
    # if BCData[group][BCVar] 
    if  BCData[group]:
        ap.setBCVar(BCVar,  BCData[group][BCVar], group)
        # ap.addDV(BCVar, familyGroup=group, name='wall_temp')

    
    nAeroNodes.append(len(X))
    XAero.extend(X)

XAero_3 = np.array(XAero)
XAero = XAero_3.flatten()

mask_aero_bottom = np.arange(0, nAeroNodes[0])
mask_aero_top = np.arange(nAeroNodes[0],nAeroNodes[0] + nAeroNodes[1])
print(CFDSolver.comm.rank, mask_aero_top, mask_aero_bottom)

#===========================================================================
# Initialize TACS
#===========================================================================
# Create the constitutvie propertes and model
props = constitutive.MaterialProperties(kappa = 230)
# con = constitutive.PlaneStressConstitutive(props)
con = constitutive.SolidConstitutive(props)
heat = elements.HeatConduction3D(con)

# Create the basis class
# quad_basis = elements.LinearQuadBasis()
basis = elements.LinearHexaBasis()

# Create the element
element = elements.Element3D(heat, basis)

# Load in the mesh
mesh = TACS.MeshLoader(comm)
# mesh.scanBDFFile('tacs_NACA0012.bdf')
mesh.scanBDFFile('n0012_hexa.bdf')

# Loop over components, creating stiffness and element object for each
num_components = mesh.getNumComponents()
for i in range(num_components):
    descriptor = mesh.getElementDescript(i)
    print('Setting element with description %s'%(descriptor))
    mesh.setElement(i, element)


# Create the assembler object
varsPerNode = heat.getVarsPerNode()
assembler = mesh.createTACS(varsPerNode)
# get structures nodes
Xpts = assembler.createNodeVec()
assembler.getNodes(Xpts)
Xpts_array = Xpts.getArray()

# get mapping of flow edge
plate_surface = []
mapping = []
opp_edge = []
Xpts_array = Xpts_array.reshape(len(Xpts_array)//3, 3)

unique_x = set(Xpts_array[:,0])
unique_x = list(unique_x)
unique_x.sort()

plate_surface = [] 
mask = []
lower_mask = []
upper_mask = []

for x in unique_x:
    mask_sec = np.where(Xpts_array[:, 0] == x)[0]
    

    # find min and max y points
    max_mask = np.where(Xpts_array[mask_sec,1] == np.max(Xpts_array[mask_sec, 1]))[0]
    min_mask = np.where(Xpts_array[mask_sec,1] == np.min(Xpts_array[mask_sec, 1]))[0]

    lower_mask.extend(mask_sec[min_mask])
    upper_mask.extend(mask_sec[max_mask])

    # mask.extend(mask_sec[min_mask], mask_sec[max_mask])
    
    # plate_surface.extend([lower_mask, upper_mask])
    # mapping.extend

lower_mask = np.array(lower_mask)
upper_mask = np.array(upper_mask)
print(MPI.COMM_WORLD.rank, lower_mask)
mask = np.concatenate((lower_mask, upper_mask))
mapping = mask
plate_surface = np.array(Xpts_array[mask])
# plate_surface_plt = plate_surface[plate_surface[:,2] == 0]
# plt.plot(Xpts_array[:,0], Xpts_array[:,1], 'o')
# plt.show()

plate_surface = plate_surface.flatten()


# Create the vectors/matrices
res = assembler.createVec()
ans = assembler.createVec()
mat = assembler.createSchurMat()
pc = TACS.Pc(mat)

# Assemble the heat conduction matrix
assembler.assembleJacobian(1.0, 0.0, 0.0, res, mat)
pc.factor()
gmres = TACS.KSM(mat, pc, 20)

# initialize MELDThermal
meld = TransferScheme.pyMELDThermal(comm, comm, 0, comm, 0, -1, 3, 0.5) #axis of symmetry, num nearest, beta
meld.setStructNodes(plate_surface)
meld.setAeroNodes(XAero)
meld.initialize()

# allocate some storage arrays
theta = np.zeros(np.sum(nAeroNodes))
# temp_check = np.zeros(np.sum(nAeroNodes))
heat= np.ones(np.sum(nAeroNodes))*0
res_holder = np.zeros(len(mapping))
delta_res = np.zeros(len(mapping))
old_res  = np.zeros(len(mapping))

ans_holder = np.zeros(len(mapping))
#===========================================================================
# Time step loop
#===========================================================================
# Iter = 0
res.zeroEntries()

Iter = 0
MaxIter = 30

n_fea = 10
n_cfd = 10

while (Iter < MaxIter):
    # set flux into TACS
    meld.transferFlux(np.array(heat), res_holder)
    # transfer flux from res holder to res array based on mapping
    # this takes the fluxes, which correspond to the upper edge of the plate
    # and places them in the residual array, which is the size of the entire plate
    # the purpose of the mapping is to place the fluxes on the nodes that correspond
    # to the upper edge of the plate

    # to make a good movie apply the delta res in increments
    print('res_holder ', np.linalg.norm(res_holder))

    delta_res = res_holder - old_res 
    
    for ii in range(1,n_fea + 1):

        res_array = res.getArray()
        res.zeroEntries()
        print('res 1', np.linalg.norm(res_array), 1/float(n_fea))
        for i in range(len(mapping)):
            res_array[mapping[i]] = old_res[i] + ii/float(n_fea)*delta_res[i]

        print('sum heat in ', np.sum(heat), 'out', np.sum(res_holder))
        print('sum heat top   ', np.sum(heat[mask_aero_top]), 'out', np.sum(res_array[upper_mask]))
        print('sum heat bottom', np.sum(heat[mask_aero_bottom]), 'out', np.sum(res_array[lower_mask]))
        print()
        if ii == 10:
            old_res = res_holder + 0.0 


        # import ipdb; ipdb.set_trace()
        print('res 2', np.linalg.norm(res_array))
        assembler.setBCs(res)
        # solve thermal problem
        print('res 3', np.linalg.norm(res_array))

        gmres.solve(res, ans)
        print('res 4', np.linalg.norm(res_array))

        assembler.setVariables(ans)
        print('res 5', np.linalg.norm(res_array))
        flag = (TACS.OUTPUT_CONNECTIVITY |
            TACS.OUTPUT_NODES |
            TACS.OUTPUT_DISPLACEMENTS |
            TACS.OUTPUT_STRAINS)

        f5 = TACS.ToFH5(assembler, TACS.SOLID_ELEMENT, flag)
        f5.writeToFile('tacs_iter_%04d.f5' % (ii + Iter*10))
        print()
    # quit()
    ans_array = ans.getArray()
    
    # get specifically the temps from the nodes in the mapping
    # i.e. the surface nodes of the structure
    for i in range(len(mapping)):
        ans_holder[i] = ans_array[mapping[i]]
    
    # transfer surface temps to theta (size of nodes on aero side)
    meld.transferTemp(ans_holder, theta)
    # print('avg temp in', np.mean(ans_holder), 'out', np.mean(theta))
    print('avg temp in', np.mean(np.array(ans_holder)), 'out', np.mean(np.array(theta)))
    print('mean temp top    in',  np.mean(ans_array[upper_mask]), 'out', np.mean(theta[mask_aero_top]))
    print('mean temp bottom in',  np.mean(ans_array[lower_mask]), 'out', np.mean(theta[mask_aero_bottom]))


    # relaxation
    fact = 0.2
    theta[mask_aero_top] = ap.BCData['heated_wall_top'][BCVar][(8,1)] + fact*( theta[mask_aero_top] - ap.BCData['heated_wall_top'][BCVar][(8,1)])
    theta[mask_aero_bottom] = ap.BCData['heated_wall_bottom'][BCVar][(7,1)] + fact*( theta[mask_aero_bottom] - ap.BCData['heated_wall_bottom'][BCVar][(7,1)])
    
    print('relaxed')
    print('mean temp top    in',  np.mean(ans_array[upper_mask]), 'out', np.mean(theta[mask_aero_top]))
    print('mean temp bottom in',  np.mean(ans_array[lower_mask]), 'out', np.mean(theta[mask_aero_bottom]))


    # set the temperatures on the surface
    ap.setBCDataArray('heated_wall_top', BCVar, theta[mask_aero_top])
    ap.setBCDataArray('heated_wall_bottom', BCVar, theta[mask_aero_bottom])
    
    # top_temp = ap.BCData['heated_wall_top'][BCVar].values()
    # top_temp[0] = 
    # ap.BCData[BCVar] += 0.1*( - ap.BCData['heated_wall_top'][BCVar])
    # ap.BCData['heated_wall_bottom'][BCVar] += 0.1*(theta[mask_aero_bottom] - ap.BCData['heated_wall_bottom'][BCVar])



    # ap.BCData['heated_wall_top'][BCVar] += 0.1*(theta[mask_aero_top] - ap.BCData['heated_wall_top'][BCVar])
    # ap.BCData['heated_wall_top'][BCVar] += 0.1*(theta[mask_aero_top] - ap.BCData['heated_wall_top'][BCVar])

    
    # ap.setDesignVars({'wall_temp_(1,1)':theta})

    # import ipdb; ipdb.set_trace()
    for _ in range(n_cfd):
        CFDSolver(ap)


    heat = np.array([])
    for group in hot_groups:
        groupHeat = CFDSolver.getHeatFluxes(groupName=group)
        heat = np.concatenate((heat,groupHeat))
        



    # print('avg heat int', np.mean(heat), 'out', np.mean(res_holder))
    # print('avg heat in top   ', np.mean(heat[mask_aero_top]), 'out', np.mean(res_array[upper_mask]))
    # print('avg heat in bottom', np.mean(heat[mask_aero_bottom]), 'out', np.mean(res_array[lower_mask]))

    # import ipdb; ipdb.set_trace()
    # set flux into assembler

    print(Iter)
    Iter += 1



# for step in range(prior_steps+1,prior_steps+1+nsteps):
#     rank = comm.Get_rank()
#     if rank == 0:
#         print('************************ ITERATION: %04d ************************'% step)

#     # Time iteration preprocessing
#     SU2Driver.Preprocess(Iter)
#     SU2Driver.BoundaryConditionsUpdate()

#     # get the normal heat flux from su2
#     for iVertex in range(nVertex_CHTMarker):
#         # if this line breaks the code, need to add the GetVertexAreaHeatFlux function
#         # to python wrapper and CDriver.
#         flux_holder[iVertex] = SU2Driver.GetVertexAreaHeatFlux(CHTMarkerID, iVertex)
#         temp_check[iVertex] = SU2Driver.GetVertexTemperature(CHTMarkerID, iVertex)
#     print('avg temp at wall = ', np.mean(np.array(temp_check)))
#     res.zeroEntries()
#     res_array = res.getArray()

#     # set flux into TACS
#     meld.transferFlux(flux_holder, res_holder)

#     if verbose:
#         print('FLUX: ', flux_holder)
#         #print('RES: ', res_holder)
#         print('Tranfered Flux on rank: ', rank)

#     res.zeroEntries()
#     res_array = res.getArray()

#     # transfer flux from res holder to res array based on mapping
#     # this takes the fluxes, which correspond to the upper edge of the plate
#     # and places them in the residual array, which is the size of the entire plate
#     # the purpose of the mapping is to place the fluxes on the nodes that correspond
#     # to the upper edge of the plate
#     for i in range(len(mapping)):
#         res_array[mapping[i]] = res_holder[i]

#     # set flux into assembler
#     assembler.setBCs(res)

#     # solve thermal problem
#     gmres.solve(res, ans)
# #assembler.setBCs(ans)
#     assembler.setVariables(ans)

#     ans_array = ans.getArray()

#     # get specifically the temps from the nodes in the mapping
#     # i.e. the surface nodes of the structure
#     for i in range(len(mapping)):
#         ans_holder[i] = ans_array[mapping[i]]
#     if verbose:        
#         print('avg temp of TACS boundary = ', np.mean(np.array(ans_holder)))

#     # transfer surface temps to theta (size of nodes on aero side)
#     meld.transferTemp(ans_holder, theta)       
#     #print('rank, Theta', rank, theta)
#     if (nVertex_CHTMarker > 0):
#         if thermal_coupling:
#             for iVertex in range(nVertex_CHTMarker):
#                 setTemp = temp_check[iVertex] + 0.5*(theta[iVertex] - temp_check[iVertex])
#                 SU2Driver.SetVertexTemperature(CHTMarkerID, iVertex, setTemp)

#             # Run an iteration of the CFD
#             SU2Driver.Run()
#             SU2Driver.Monitor(Iter)
#             SU2Driver.Output(Iter)

#             Iter += 1
#             delta_avg_temp = abs(np.mean(np.array(ans_holder)) - np.mean(np.array(temp_check)))

#     if (step % 10 ==0):
#         flag = (TACS.OUTPUT_CONNECTIVITY |
#             TACS.OUTPUT_NODES |
#             TACS.OUTPUT_DISPLACEMENTS |
#             TACS.OUTPUT_STRAINS)
#         f5 = TACS.ToFH5(assembler, TACS.SCALAR_2D_ELEMENT, flag)
#         f5.writeToFile('tacs_iter_%04d.f5' % (step))
#     print('Delta Temperature: ')
#     print(delta_avg_temp)
# #===========================================================================
# # Finish the tacs analysis
# #===========================================================================
# # Set the element flag
# flag = (TACS.OUTPUT_CONNECTIVITY |
#         TACS.OUTPUT_NODES |
#         TACS.OUTPUT_DISPLACEMENTS |
#         TACS.OUTPUT_STRAINS)
# f5 = TACS.ToFH5(assembler, TACS.SCALAR_2D_ELEMENT, flag)
# f5.writeToFile('tacs_flatplate.f5')

# #===========================================================================
# # Finish the SU2 analysis.
# #===========================================================================
# SU2Driver.Postprocessing()
