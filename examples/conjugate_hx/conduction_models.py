"""conduction models for conjugate heat transfer problem """
import numpy as np
import openmdao.api as om
from mpi4py import MPI
import copy

from mphys.base_classes import SolverObjectBasedSystem

class Conduction(om.ExplicitComponent, SolverObjectBasedSystem):
    """
    Analytic 1D Conduction Model
    """


    def initialize(self):
        self.options.declare('solver_options')
 
        self.solver_objects = {'Adflow':None}
        # self.options['distributed'] = True
        self.solvers_init = False


    #would be removed after PEOM 22
    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver

        self.solvers_init = True

    def setup(self):

        
        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        solver = self.solver =self.solver_objects['Adflow']
        

        x_a = solver.getSurfaceCoordinates().flatten(order='C')

        local_coord_size = solver.getSurfaceCoordinates().size

        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])
        self.n1 = int(n1)
        self.n2 = int(n2)

        # self.add_input('x_a', src_indices=np.arange(n1,n2,dtype=int),shape=local_coord_size, val=x_a)
        self.add_input('x_a', shape_by_conn=True)




        # Global Design Variable
        self.add_input('Q', val=np.array([-200]), units='W/m**2')
        self.add_output('T_surf', val=np.array([[273.0 + 163]]) , units='K')


        self.K_Al = 120  # W/(m*K) (thermal conductivity of alluminium )
        self.K_epoxy = 1 # W/(m*K) (thermal conductivity of epoxy )
        self.motor_radius = 0.15645/2 #m 
        self.epoxy_thickness = 1e-4 # m
        self.T_mid = 273.0 + 165  # deg kelvin ~220 degs C

    def compute(self, inputs, outputs):
        """
            solves for the wall temperature given the thermal conductivity of
            the material and the temperature at the center
        """

        x_a = inputs['x_a']
        # 4 blocks, nodes per block, 3 directions
        x_a = np.reshape(x_a, (len(x_a)//3, 3))

        # get just the surface nodes on the motor
        # x_motor  = self.solver.mapVector(x_a, self.solver.allWallsGroup, self.solver.allIsothermalWallsGroup)
        x_motor = x_a

        
        # 4 blocks, nodes per block, 3 directions
        nNodes_motor = self.comm.allreduce(x_motor.shape[0], op=MPI.SUM)

        if x_motor.shape[0] > 0 :

            nBlks = int(float(x_motor.shape[0])/nNodes_motor*4)

            x_motor = np.reshape(x_motor, (nBlks, x_motor.size//(3*nBlks), 3))
            R_therm_cells, _ = self.getThermalResistance(x_motor)
            R_inv = np.sum(R_therm_cells**-1)
        else:
            R_inv = np.array([0.0])
        
        
        R_inv_sum = self.comm.allreduce(R_inv, op=MPI.SUM)
        R_therm_eqiv = R_inv_sum**-1 
        T_surf = self.T_mid + inputs['Q']*R_therm_eqiv 
        outputs['T_surf'] = T_surf
        if self.comm.rank == 0:
            print('=====================')
            print(self.comm.rank, 'T_Surf', outputs['T_surf'], self.T_mid, inputs['Q'], R_therm_eqiv,  inputs['Q']*R_therm_eqiv)
            print('=====================')

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # print('yes')
        if mode == 'fwd':
            if 'T_surf' in d_outputs:
                x_a = inputs['x_a']
                d_x_a = d_inputs['x_a']

                x_a = np.reshape(x_a, (len(x_a)//3, 3))
                d_x_a = np.reshape(d_x_a, (len(d_x_a)//3, 3))
                              
                x_motor  = self.solver.mapVector(x_a, self.solver.allWallsGroup, self.solver.allIsothermalWallsGroup)
                d_x_motor  = self.solver.mapVector(d_x_a, self.solver.allWallsGroup, self.solver.allIsothermalWallsGroup)

                self.comm.barrier()
                nNodes_motor = self.comm.allreduce(x_motor.shape[0], op=MPI.SUM)

                if x_motor.shape[0] > 0 :

                    nBlks = int(float(x_motor.shape[0])/nNodes_motor*4)

                    print(self.comm.rank, x_motor.shape, nBlks)
                    x_motor = np.reshape(x_motor, (nBlks, x_motor.size//(3*nBlks), 3))
                    d_x_motor = np.reshape(d_x_motor, (nBlks, d_x_motor.size//(3*nBlks), 3))
                    R_therm_cells, _, d_R_therm_cells, _ = self.getThermalResistance_d(x_motor, d_x_motor)
                    # if not np.sum(d_R_therm_cells) == 0.0:
                    #     print('d_R_therm_cells', d_R_therm_cells)
                    R_inv = np.sum(R_therm_cells**-1)
                    d_R_inv = np.sum(-1*R_therm_cells**-2*d_R_therm_cells)
                else:
                    R_inv = np.array([0.0])
                    d_R_inv = np.array([0.0])
                
                self.comm.barrier()
                R_inv_sum = self.comm.allreduce(R_inv, op=MPI.SUM)
                d_R_inv_sum = self.comm.allreduce(d_R_inv, op=MPI.SUM)


                R_therm_eqiv = R_inv_sum**-1 
                d_R_therm_eqiv = -1*R_inv_sum**-2 * d_R_inv_sum 

                d_T_surf = d_inputs['Q']*R_therm_eqiv  + inputs['Q']*d_R_therm_eqiv

                d_outputs['T_surf'] = d_T_surf

        if mode == 'rev':
            if 'T_surf' in d_outputs:
                x_a = inputs['x_a']
                x_a = np.reshape(x_a, (len(x_a)//3, 3))
                              
                x_motor  = self.solver.mapVector(x_a, self.solver.allWallsGroup, self.solver.allIsothermalWallsGroup)

                nNodes_motor = self.comm.allreduce(x_motor.shape[0], op=MPI.SUM)
                print(self.comm.rank, x_motor.shape[0])
                if x_motor.shape[0] > 0 :
                    hasMotorNodes = True

                    nBlks = int(float(x_motor.shape[0])/nNodes_motor*4)

                    x_motor = np.reshape(x_motor, (nBlks, x_motor.size//(3*nBlks), 3))
                    R_therm_cells, _ = self.getThermalResistance(x_motor)
                    R_inv = np.sum(R_therm_cells**-1)
                else:
                    hasMotorNodes = False
                    R_inv = np.array([0.0])
                
                R_inv_sum = self.comm.allreduce(R_inv, op=MPI.SUM)
                R_therm_eqiv = R_inv_sum**-1 

                d_T_surf = d_outputs['T_surf']  

                d_R_therm_eqiv = inputs['Q']*d_T_surf
                if 'Q' in d_inputs:
                    d_inputs['Q'] += np.sum(d_T_surf*R_therm_eqiv )

                d_R_inv_sum = -1*R_inv_sum**-2 * d_R_therm_eqiv 
                d_R_inv = d_R_inv_sum

                if hasMotorNodes :

                    nCells_blk = int((np.sqrt(x_motor.shape[1]) - 1)**2)
                    nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

                    d_area_cells = np.zeros((nBlks,nCells_blk))
                    d_R_therm_cells = np.zeros((nBlks,nCells_blk))

                    d_R_therm_cells += -1*R_therm_cells**-2*d_R_inv
                    d_R_inv = np.sum(-1*R_therm_cells**-2*d_R_therm_cells)

                    d_x_motor = self.getThermalResistance_b(x_motor, d_R_therm_cells, d_area_cells)
                
                    d_x_a  = self.solver.mapVector(d_x_motor, self.solver.allIsothermalWallsGroup, self.solver.allWallsGroup)
                    if 'x_a' in d_inputs:
                        d_inputs['x_a'] += d_x_a.flatten(order='C')
                
    def getThermalResistance(self, x_motor):
        """
            calculates the thermal resistance for each cell for the 1D conduction
            analyisis

            x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes


        
        """

        #number of cells on  each block face
        nNodes_row = int(np.sqrt(x_motor.shape[1])) # the cells are always perfect squares
        nCells_blk = int((nNodes_row - 1)**2)
        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

        R_therm_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
        area_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
        
        # radial distance from the x-axis
        radius_nodes = np.sqrt(x_motor[:,:, 1]**2 + x_motor[:,:, 2]**2)

        # calculate the thermal resistance for each cell
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                offset = idx_cell + (idx_cell)//nCells_row
                # calculate the radius of the midpoint of the cell based on the 
                # radi of the nodes 
                radius_cell = 0.25*(radius_nodes[idx_blk,0 + offset] +
                                    radius_nodes[idx_blk,1 + offset] + 
                                    radius_nodes[idx_blk,0 + nNodes_row + offset] + 
                                    radius_nodes[idx_blk,1 + nNodes_row + offset])


                # calculate the area of a cell

                x = x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]
                # import ipdb; ipdb.set_trace()

                area_cells[idx_blk, idx_cell] = getCellArea(x)

                # add the thermal resistance of the heat sink and epoxy used to attach it
                # (in series so the resistances add)
                R_therm_epoxy = self.epoxy_thickness/(self.K_epoxy*area_cells[idx_blk, idx_cell])
                R_therm_Al = (radius_cell -self.motor_radius)/(self.K_Al*area_cells[idx_blk, idx_cell])          
                R_therm_cells[idx_blk, idx_cell] =  R_therm_epoxy + R_therm_Al
                                   
        # import ipdb; ipdb.set_trace()
        return R_therm_cells, area_cells

    def getThermalResistance_d(self, x_motor, d_x_motor):
        """
            forward mode AD
            calculates the thermal resistance for each cell for the 1D conduction
            analyisis

            x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes


            d_x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes derivative seed
      
        """

        #number of cells on  each block face
        nNodes_row = int(np.sqrt(x_motor.shape[1])) # the nodes are always perfect squares

        nCells_blk = int((np.sqrt(x_motor.shape[1]) - 1)**2)
        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

        R_therm_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
        d_R_therm_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
        area_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
        d_area_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
       

        # radial distance from the x-axis
        radius_nodes = np.sqrt(x_motor[:,:, 1]**2 + x_motor[:,:, 2]**2)
        d_radius_nodes = 1/(2*radius_nodes)*(2*x_motor[:,:, 1]*d_x_motor[:,:,1] +
                                             2*x_motor[:,:, 2]*d_x_motor[:,:,2])
        
        # calculate the thermal resistance for each cell
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                offset = idx_cell + (idx_cell)//nCells_row
                # calculate the radius of the midpoint of the cell based on the 
                # radi of the nodes 
                radius_cell = 0.25*(radius_nodes[idx_blk,0 + offset] +
                                    radius_nodes[idx_blk,1 + offset] + 
                                    radius_nodes[idx_blk,0 + nNodes_row + offset] + 
                                    radius_nodes[idx_blk,1 + nNodes_row + offset])

                d_radius_cell = 0.25*(d_radius_nodes[idx_blk,0 + offset] +
                                      d_radius_nodes[idx_blk,1 + offset] + 
                                      d_radius_nodes[idx_blk,0 + nNodes_row + offset] + 
                                      d_radius_nodes[idx_blk,1 + nNodes_row + offset])


                # move the nodes so the first is now at the origin 
                x = x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]
                d_x = d_x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]

                area_cells[idx_blk, idx_cell], d_area_cells[idx_blk, idx_cell] = getCellArea_d(x, d_x)
                # add the thermal resistance of the heat sink and epoxy used to attach it
                # (in series so the resistances add)
                
                R_therm_epoxy = self.epoxy_thickness/(self.K_epoxy*area_cells[idx_blk, idx_cell])
                R_therm_Al = (radius_cell -self.motor_radius)/(self.K_Al*area_cells[idx_blk, idx_cell])          
                R_therm_cells[idx_blk, idx_cell] =  R_therm_epoxy + R_therm_Al
                
                d_R_therm_epoxy = - self.epoxy_thickness/(self.K_epoxy*area_cells[idx_blk,idx_cell]**2) * d_area_cells[idx_blk,idx_cell] 
                d_R_therm_Al = (d_radius_cell)/(self.K_Al*area_cells[idx_blk,idx_cell]) + \
                              -(radius_cell -self.motor_radius)/(self.K_Al*area_cells[idx_blk,idx_cell]**2) * d_area_cells[idx_blk,idx_cell]          

                d_R_therm_cells[idx_blk, idx_cell] =  d_R_therm_epoxy + d_R_therm_Al

        return R_therm_cells, area_cells,  d_R_therm_cells, d_area_cells

    def getThermalResistance_b(self, x_motor, d_R_therm_cells, d_area_cells):
        """
            reverse mode AD
            calculates the thermal resistance for each cell for the 1D conduction
            analyisis

            x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes


            d_x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes derivative seed
      
        """
        d_x_motor = np.zeros(x_motor.shape)
        _, area_cells = self.getThermalResistance(x_motor)


        nNodes_row = int(np.sqrt(x_motor.shape[1])) # the nodes are always perfect squares

        #number of cells on  each block face
        nCells_blk = int((np.sqrt(x_motor.shape[1]) - 1)**2)
        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

        radius_nodes = np.sqrt(x_motor[:,:, 1]**2 + x_motor[:,:, 2]**2)
        d_radius_nodes = np.zeros((x_motor.shape[0],x_motor.shape[1]))

        d_x = np.zeros((4)) # there will always be four nodes per cell


        # calculate the thermal resistance for each cell
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                offset = idx_cell + (idx_cell)//nCells_row

                radius_cell = 0.25*(radius_nodes[idx_blk,0 + offset] +
                                    radius_nodes[idx_blk,1 + offset] + 
                                    radius_nodes[idx_blk,0 + nNodes_row + offset] + 
                                    radius_nodes[idx_blk,1 + nNodes_row + offset])

                x = x_motor[idx_blk, np.array([0,1,3,4])+ offset]
                x = x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]

                d_R_therm_epoxy = d_R_therm_cells[idx_blk, idx_cell]
                d_R_therm_Al = d_R_therm_cells[idx_blk, idx_cell]
                


                d_radius_cell = d_R_therm_Al/(self.K_Al*area_cells[idx_blk, idx_cell])

                d_area_cells[idx_blk, idx_cell] += -(radius_cell -self.motor_radius)/(self.K_Al*area_cells[idx_blk, idx_cell]**2) * d_R_therm_Al \
                                        - self.epoxy_thickness/(self.K_epoxy*area_cells[idx_blk,idx_cell]**2) * d_R_therm_epoxy
                
                d_x = getCellArea_b(x, d_area_cells[idx_blk, idx_cell])

                d_x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += d_x 

                 # calculate the area of a cell

                d_radius_nodes[idx_blk,np.array([0,1,0+nNodes_row,1 + nNodes_row]) + offset] += 0.25*d_radius_cell
                # radi of the nodes 

        d_x_motor[:,:,1] += 1/(2*radius_nodes)*(2*x_motor[:,:, 1]*d_radius_nodes)
        d_x_motor[:,:,2] += 1/(2*radius_nodes)*(2*x_motor[:,:, 2]*d_radius_nodes)
          
        

        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares
        nCells_blk = int((np.sqrt(x_motor.shape[1]) - 1)**2)
        #number of cells on  each block face
        
        return d_x_motor


class ConductionNodal(om.ExplicitComponent):
    """
    Nodal Analytic 1D Conduction Model
    """

    def initialize(self):
        # self.options.declare('get_solver')
        self.options['distributed'] = True

    def setup(self):

        self.add_input('x_a', shape_by_conn=True)
        self.add_input('heatflux', shape_by_conn=True , units='W/m**2')
        self.add_output('T_surf', copy_shape='heatflux', units='K')
        
        self.K_Al = 120  # W/(m*K) (thermal conductivity of alluminium )
        self.K_epoxy = 1 # W/(m*K) (thermal conductivity of epoxy )
        self.motor_radius = 0.15645/2 #m 
        self.epoxy_thickness = 1e-4 # m

    def determine_shape(self):
        self.copy_var_shape('heatflux', 'T_surf')


    def getTempInner(self, x_motor):
        """ given a set of motor points calculate the temperature of inner side of the heat sink"""
        temp = x_motor[:,:,0]**2*-23846.0 + x_motor[:,:,0]*11780.0  - 1281  + 273
        return temp

    def getTempInner_d(self, x_motor, d_x_motor):
        """ given a set of motor points calculate the temperature of inner side of the heat sink"""
        temp = x_motor[:,:,0]**2*-23846.0 + x_motor[:,:,0]*11780.0  - 1281  + 273
        d_temp = 2*x_motor[:,:,0]*-23846.0*d_x_motor[:,:,0] + d_x_motor[:,:,0]*11780.0 
        return temp, d_temp


    def getTempInner_b(self, x_motor, d_temp):
        """ given a set of motor points calculate the temperature of inner side of the heat sink"""
        d_x_motor = np.zeros(x_motor.shape)
        d_x_motor[:,:,0] = 2*x_motor[:,:,0]*-23846.0*d_temp + d_temp*11780.0 
        return d_x_motor


    # def getTempInner(self, x_motor):
    #     """ given a set of motor points calculate the temperature of inner side of the heat sink"""
    #     temp = 500*(x_motor[:,:,0]- 0.213) +  273 + 60
        
    #     temp[0] += 20
    #     temp[1] += 30
    #     temp[3] += 40
    #     temp[2] += 50

    #     return temp


    def compute(self, inputs, outputs):
        """
            solves for the wall temperature given the thermal conductivity of
            the material and the temperature at the center
        """
        print('========== compute ===========')
        # import pdb; pdb.set_trace()
        x_a = inputs['x_a']
        heatflux_local =  np.reshape(inputs['heatflux'], (4, len(inputs['heatflux'])//4))

        # 4 blocks, nodes per block, 3 directions
        x_a = np.reshape(x_a, (x_a.size//3, 3))

        x_motor = x_a

        
        x_motor_global = np.reshape(x_motor, (4, x_motor.size//(3*4), 3))
        
        # ------------------------ 
        R_therm, area_nodes = self.getNodalThermalResistance(x_motor_global)
        
        T_mid = self.getTempInner(x_motor_global)
        
        T_surf = T_mid + (heatflux_local*area_nodes)*R_therm


        T_surf =  T_surf.flatten()
        print(T_surf)
        t_list = self.comm.allgather(T_surf.size)
        irank = self.comm.rank
        print(irank, t_list)
        self.t1 = int(np.sum(t_list[:irank]))
        self.t2 = int(np.sum(t_list[:irank+1]))
        outputs['T_surf'] = T_surf[self.t1:self.t2]
        

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # print('yes')
        if mode == 'fwd':
            print('========== fwd ===========')

            if 'T_surf' in d_outputs:
                x_a = inputs['x_a']
                heatflux_local = inputs['heatflux']
                
                if 'x_a' in d_inputs:
                    d_x_a = d_inputs['x_a']
                else:
                    d_x_a = np.zeros(x_a.shape)
                

                x_a = np.reshape(x_a, (x_a.size//3, 3))
                d_x_a = np.reshape(d_x_a, (d_x_a.size//3, 3))

                x_motor = x_a
                d_x_motor = d_x_a

                if 'heatflux' in d_inputs:
                    d_heatflux_local = d_inputs['heatflux']
                else:
                    d_heatflux_local = np.zeros((x_motor.shape[0]))
                # =============== get global variables =====================

                x_motor_global = np.reshape(x_motor, (4, x_motor.size//(3*4), 3))
                d_x_motor_global = np.reshape(d_x_motor, (4, d_x_motor.size//(3*4), 3))




                heatflux_global = np.reshape(heatflux_local, (4, heatflux_local.size//(4)))
                d_heatflux_global = np.reshape(d_heatflux_local, (4, d_heatflux_local.size//(4)))




                R_therm, area_nodes, d_R_therm, d_area_nodes = self.getNodalThermalResistance_d(x_motor_global, d_x_motor_global)
                # _, _, d_R_therm_dot, d_area_nodes_dot = self.getNodalThermalResistance_d(x_motor_global, d_x_motor_global_dot)
                
                T_mid, d_T_mid = self.getTempInner_d(x_motor_global, d_x_motor_global)
            
                d_T_surf = d_T_mid + d_heatflux_global*area_nodes*R_therm \
                                   + heatflux_global*d_area_nodes*R_therm \
                                   + heatflux_global*area_nodes*d_R_therm
                
                # print('d_heatflux_global*area_nodes*R_therm',d_heatflux_global,area_nodes,R_therm)

                d_T_surf =  d_T_surf.flatten()
                d_outputs['T_surf'] = d_T_surf[self.t1:self.t2]
                print(self.comm.rank, 'd_T_surf', d_outputs['T_surf'], self.t1, self.t2)

        if mode == 'rev':
            # pass
            # raise Exception
            # import pdb; pdb.set_trace()
            print('================ rev ================')
    

            x_a = inputs['x_a']
            heatflux_local = inputs['heatflux']

            x_a = np.reshape(x_a, (x_a.size//3, 3))
            x_motor = x_a

            x_motor_global = np.reshape(x_motor, (4, x_motor.size//(3*4), 3))
            heatflux_global = np.reshape(heatflux_local, (4, heatflux_local.size//(4)))

            # ------------------------ 
            R_therm, area_nodes_global = self.getNodalThermalResistance(x_motor_global)
            

            if 'T_surf' in d_outputs:
                d_T_surf = np.zeros(area_nodes_global.size)
                d_T_surf[self.t1:self.t2] = d_outputs['T_surf']  

                #comunnicate to form global temperature matrix

                # d_outputs['T_surf'] = d_T_surf[self.t1:self.t2]

                print(self.comm.rank, d_T_surf)
                # d_T_surf = self.comm.allreduce(d_T_surf, op=MPI.SUM)
                d_T_surf = np.reshape(d_T_surf,area_nodes_global.shape)

                # d_T_surf =  d_T_surf.flatten()

                # if 'heatflux' in d_inputs:
                #         d_heatflux_local = d_inputs['heatflux']
                #     else:
                #         d_heatflux_local = np.zeros((x_motor.shape[0]))

                # print('d_heatflux_global*area_nodes_global*R_therm',d_heatflux_global,area_nodes_global,R_therm)
                d_T_mid = d_T_surf
                if 'heatflux' in d_inputs:
                    d_heatflux_global = d_T_surf*area_nodes_global*R_therm
                    d_inputs['heatflux'] = d_heatflux_global.flatten()[self.t1:self.t2]
                    print(self.comm.rank,  'd_heat', d_heatflux_global.flatten()[self.t1:self.t2], self.t1, self.t2)

                
                d_area_nodes_global = heatflux_global*d_T_surf*R_therm 
                d_R_therm = heatflux_global*area_nodes_global*d_T_surf

                d_x_motor_global = self.getTempInner_b(x_motor_global, d_T_mid)
                
                d_x_motor_global += self.getNodalThermalResistance_b(x_motor_global, area_nodes_global,\
                                                                        d_R_therm, d_area_nodes_global)


                d_x_a  = np.reshape(d_x_motor_global , x_a.shape).flatten()
                if 'x_a' in d_inputs:
                    d_inputs['x_a'] += d_x_a
                    print(self.comm.rank, 'd_inputs[x_a]', d_x_a, self.t1, self.t2)
                    # import pdb; pdb.set_trace()
                
                # import pdb; pdb.set_trace()
         


                
    def getNodalThermalResistance(self, x_motor):
        """
            calculates the thermal resistance for each cell for the 1D conduction
            analyisis

            x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes

            steps

            0) get radius at mesh points 
            1)get area at the nodes by adding a 1/4 of each nodal value 
            2) calc resistance at nodes 

        
        """

        #number of cells on  each block face
        nNodes_row = int(np.sqrt(x_motor.shape[1])) # the cells are always perfect squares
        if nNodes_row >1:
            nCells_blk = int((nNodes_row - 1)**2)
        else:
            nCells_blk = 0
        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

        # area_cells = np.zeros((x_motor.shape[0],nCells_blk), dtype=x_motor.dtype)
        area_nodes = np.zeros((x_motor.shape[0], x_motor.shape[1]), dtype=x_motor.dtype)
        R_therm_nodes = np.zeros((x_motor.shape[0], x_motor.shape[1]), dtype=x_motor.dtype)
        R_therm_epoxy = np.zeros((x_motor.shape[0], x_motor.shape[1]), dtype=x_motor.dtype)
        R_therm_Al = np.zeros((x_motor.shape[0], x_motor.shape[1]), dtype=x_motor.dtype)
        
        # radial distance from the x-axis
        radius_nodes = np.sqrt(x_motor[:,:, 1]**2 + x_motor[:,:, 2]**2)
        
        # calculate the area for cell and add it to the nodes
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                offset = idx_cell + (idx_cell)//nCells_row
                x = x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]

                area_cell = getCellArea(x)

                area_nodes[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += area_cell/4



        area_nodes_global = copy.copy(area_nodes)
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                idx = np.argwhere(np.all(x_motor == x_motor[idx_blk, idx_cell], axis=-1))

                #add the nodes together
                if idx.shape[0] > 1:
                    area_nodes_global[idx[:,0],idx[:,1]] = np.sum(area_nodes[idx[:,0],idx[:,1]])


        # print(self.comm.rank, 'area', area_nodes_global, x_motor)
        # import ipdb; ipdb.set_trace()
        R_therm_epoxy = self.epoxy_thickness/(self.K_epoxy*area_nodes_global)
        R_therm_Al = (radius_nodes -self.motor_radius)/(self.K_Al*area_nodes_global)          
        
        R_therm_nodes =   R_therm_epoxy + R_therm_Al
        # print(self.comm.rank, R_therm_nodes)
        # print(self.comm.rank, x_motor)

        return R_therm_nodes, area_nodes_global

    def getNodalThermalResistance_d(self, x_motor, d_x_motor):
        """
            calculates the thermal resistance for each cell for the 1D conduction
            analyisis

            x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes

            steps

            0) get radius at mesh points 
            1)get area at the nodes by adding a 1/4 of each nodal value 
            2) calc resistance at nodes 

        
        """

        #number of cells on  each block face
        nNodes_row = int(np.sqrt(x_motor.shape[1])) # the cells are always perfect squares
        nCells_blk = int((nNodes_row - 1)**2)
        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

        area_nodes = np.zeros((x_motor.shape[0], x_motor.shape[1]), dtype=x_motor.dtype)
        d_area_nodes = np.zeros((x_motor.shape[0], x_motor.shape[1]), dtype=x_motor.dtype)

        
        # radial distance from the x-axis
        radius_nodes = np.sqrt(x_motor[:,:, 1]**2 + x_motor[:,:, 2]**2)
        d_radius_nodes = 1/(2*radius_nodes)*(2*x_motor[:,:, 1]*d_x_motor[:,:,1] +
                                        2*x_motor[:,:, 2]*d_x_motor[:,:,2])

        # calculate the area for cell and add it to the nodes
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                offset = idx_cell + (idx_cell)//nCells_row
                x = x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]
                d_x = d_x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]
                # import ipdb; ipdb.set_trace()

                area_cell, d_area_cell = getCellArea_d(x, d_x)

                area_nodes[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += area_cell/4
                d_area_nodes[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += d_area_cell/4


        # correct for the connenctivity of the blocks 

        area_nodes_global = copy.copy(area_nodes)
        d_area_nodes_global = copy.copy(d_area_nodes)
        
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                idx = np.argwhere(np.all(x_motor == x_motor[idx_blk, idx_cell], axis=-1))

                #add the nodes together
                if idx.shape[0] > 1:
                    area_nodes_global[idx[:,0],idx[:,1]] = np.sum(area_nodes[idx[:,0],idx[:,1]])
                    d_area_nodes_global[idx[:,0],idx[:,1]] = np.sum(d_area_nodes[idx[:,0],idx[:,1]])

        R_therm_epoxy = self.epoxy_thickness/(self.K_epoxy*area_nodes_global)
        R_therm_Al = (radius_nodes -self.motor_radius)/(self.K_Al*area_nodes_global)          
        R_therm_nodes =   R_therm_epoxy + R_therm_Al

        d_R_therm_epoxy = - self.epoxy_thickness/(self.K_epoxy*area_nodes_global**2) * d_area_nodes_global 
        d_R_therm_Al = (d_radius_nodes)/(self.K_Al*area_nodes_global) + \
                        -(radius_nodes -self.motor_radius)/(self.K_Al*area_nodes_global**2) * d_area_nodes_global          

        d_R_therm_nodes =  d_R_therm_epoxy + d_R_therm_Al

        # import ipdb; ipdb.set_trace()
        return R_therm_nodes, area_nodes_global, d_R_therm_nodes, d_area_nodes_global


    def getNodalThermalResistance_b(self, x_motor, area_nodes_global,  d_R_therm, d_area_nodes_global):
        """
            reverse mode AD
            calculates the thermal resistance for each cell for the 1D conduction
            analyisis

            x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes


            d_x_motor: np.array(4, nNodes_blk, 3)
            x, y, z coordinates of the motor surface nodes derivative seed
      
        """
        d_x_motor = np.zeros(x_motor.shape)
        # _, area_nodal_global = self.getNodalThermalResistance(x_motor)


        #######

        nNodes_row = int(np.sqrt(x_motor.shape[1])) # the nodes are always perfect squares

        #number of cells on  each block face
        if nNodes_row >1:
            nCells_blk = int((nNodes_row - 1)**2)
        else:
            nCells_blk = 0
        # nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares

        nCells_row = int(np.sqrt(nCells_blk)) # the cells are always perfect squares


        radius_nodes = np.sqrt(x_motor[:,:, 1]**2 + x_motor[:,:, 2]**2)
        # d_radius_nodes = np.zeros((x_motor.shape[0],x_motor.shape[1]))

        # d_x = np.zeros((4)) # there will always be four nodes per cell
        d_R_therm_epoxy = d_R_therm
        d_R_therm_Al = d_R_therm

        d_area_nodes_tmp = copy.copy(d_area_nodes_global)
        d_radius_nodes = d_R_therm_Al/(self.K_Al*area_nodes_global)
        d_area_nodes_tmp  +=  -(radius_nodes -self.motor_radius)/(self.K_Al*area_nodes_global**2) * d_R_therm_Al          
        d_area_nodes_tmp  += - self.epoxy_thickness/(self.K_epoxy*area_nodes_global**2) * d_R_therm_epoxy 
        d_area_nodes = copy.copy(d_area_nodes_tmp)
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                idx = np.argwhere(np.all(x_motor == x_motor[idx_blk, idx_cell], axis=-1))

                if idx.shape[0] > 1:
                    d_area_nodes[idx[:,0],idx[:,1]] = np.sum(d_area_nodes_tmp[idx[:,0],idx[:,1]])
                    #  = np.sum(d_area_nodes[idx[:,0],idx[:,1]])
                    # d_area_nodes[idx[0,0],idx[0,1]] = d_area_nodes_tmp[idx[0,0],idx[0,1]]
                    # d_area_nodes[idx[1,0],idx[1,1]] = d_area_nodes_tmp[idx[0,0],idx[0,1]]
                #add the nodes together

        
        # correct for the connenctivity of the blocks 
        for idx_blk in range(x_motor.shape[0]):
            for idx_cell in range(nCells_blk):
                offset = idx_cell + (idx_cell)//nCells_row
                x = x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset]

                # area_nodes[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += area_cell/4
                # d_area_nodes[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += d_area_cell/4
                d_area_cell = np.sum(d_area_nodes[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset])/4
                d_x = getCellArea_b(x, d_area_cell)

                # import ipdb; ipdb.set_trace()
                d_x_motor[idx_blk, np.array([0,1,0+nNodes_row,1 + nNodes_row])+ offset] += d_x

        # calculate the area for cell and add it to the nodes

        d_x_motor[:,:,1] += 1/(2*radius_nodes)*(2*x_motor[:,:, 1]*d_radius_nodes)
        d_x_motor[:,:,2] += 1/(2*radius_nodes)*(2*x_motor[:,:, 2]*d_radius_nodes)
          
        return d_x_motor





### cell area functions


def getAreaTriangle(x1, x2):
    i = x1[1]*x2[2]- x1[2]*x2[1]
    j = x1[0]*x2[2] - x1[2]*x2[0]
    k = x1[0]*x2[1] - x1[1]*x2[0]
    area =  0.5*(np.sqrt(i**2 + j**2 + k**2))
    return area

def getAreaTriangle_d(x1, x2, d_x1, d_x2):
    i =             x1[1]*x2[2]               - x1[2]*x2[1]
    d_i = d_x1[1]*x2[2] + x1[1]*d_x2[2] - d_x1[2]*x2[1] - x1[2]*d_x2[1]
    j =            x1[0]*x2[2]                - x1[2]*x2[0]
    d_j = d_x1[0]*x2[2] + x1[0]*d_x2[2] - d_x1[2]*x2[0] - x1[2]*d_x2[0]
    k =            x1[0]*x2[1]                - x1[1]*x2[0]
    d_k = d_x1[0]*x2[1] + x1[0]*d_x2[1] - d_x1[1]*x2[0] - x1[1]*d_x2[0]

    area = 0.5*(np.sqrt(i**2 + j**2 + k**2))
    d_area = 0.125/area*(2*i*d_i + 2*j*d_j + 2*k*d_k)
    
    return area , d_area

def getAreaTriangle_b(x1, x2, d_area):
    d_x1 = np.zeros(x1.shape)
    d_x2 = np.zeros(x1.shape)
    
    i = x1[1]*x2[2]- x1[2]*x2[1]
    j = x1[0]*x2[2] - x1[2]*x2[0]
    k = x1[0]*x2[1] - x1[1]*x2[0]
    area =  0.5*(np.sqrt(i**2 + j**2 + k**2))

    d_i = 0.125/area*(2*i*d_area)
    d_j = 0.125/area*(2*j*d_area)
    d_k = 0.125/area*(2*k*d_area)
    
    d_x1[0] +=  d_k*x2[1] 
    d_x2[1] +=  x1[0]*d_k 
    d_x1[1] += -d_k*x2[0] 
    d_x2[0] += -x1[1]*d_k

    d_x1[0] +=  d_j*x2[2] 
    d_x2[2] +=  x1[0]*d_j 
    d_x1[2] += -d_j*x2[0] 
    d_x2[0] += -x1[2]*d_j

    d_x1[1] +=  d_i*x2[2] 
    d_x2[2] +=  x1[1]*d_i 
    d_x1[2] += -d_i*x2[1] 
    d_x2[1] += -x1[2]*d_i

    return d_x1, d_x2

def getCellArea(x):
    # move the nodes so the first is now at the origin 
    x -= x[0]

    # find the area of both triangles and add then together
    # same as  (||X1 x X3|| + ||X2 x X3||)/2
            
    area_tri_1 = getAreaTriangle(x[1], x[3])
    area_tri_2 = getAreaTriangle(x[2], x[3])

    return area_tri_1 + area_tri_2

def getCellArea_d(x, d_x):
    x -= x[0]
    d_x -= d_x[0]

    area_tri_1, d_area_tri_1 = getAreaTriangle_d(x[1], x[3], d_x[1], d_x[3])
    area_tri_2, d_area_tri_2 = getAreaTriangle_d(x[2], x[3], d_x[2], d_x[3])

    return area_tri_1 + area_tri_2, d_area_tri_1 + d_area_tri_2

def getCellArea_b(x, d_area):
    d_x = np.zeros(x.shape)
    tmp_x = np.zeros(x.shape)
    
    
    x -= x[0]
    

    tmp_x[1], tmp_x[3] = getAreaTriangle_b( x[1], x[3], d_area)  
    tmp_x[2], tmp3 = getAreaTriangle_b( x[2], x[3], d_area)  
    tmp_x[3] += tmp3

    d_x += tmp_x
    # some tricky business becuase of how x -= x[0] works
    d_x[0] -= tmp_x[0] + tmp_x[1] + tmp_x[2] + tmp_x[3] 

    return d_x


