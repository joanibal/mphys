import openmdao.api as om
from pygeo import DVGeometry, DVConstraints
from mpi4py import MPI
import numpy as np
import os 

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('ffd_file', allow_none=False)
        self.options['distributed'] = True

    def setup(self):
        # create the DVGeo object that does the computations
        ffd_file = self.options['ffd_file']
        self.DVGeo = DVGeometry(ffd_file)

    def compute(self, inputs, outputs):

        # inputs are the geometric design variables
        self.DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptSet in self.DVGeo.points:
            # update this pointset and write it as output
            outputs[ptSet] = self.DVGeo.update(ptSet).flatten()

    def nom_addPointSet(self, points, ptName, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points)//3, 3), ptName, **kwargs)

        # add an output to the om component
        self.add_output(ptName, val=points.flatten())

    def nom_add_point_dict(self, point_dict):
        # add every pointset in the dict, and set the ptset name as the key
        for k,v in point_dict.items():
            self.nom_addPointSet(v, k)

    def nom_addGeoDVGlobal(self, dvName, value, func):
        # define the input
        self.add_input(dvName, shape=value.shape)

        # call the dvgeo object and add this dv
        self.DVGeo.addGeoDVGlobal(dvName, value, func)

    def nom_addRefAxis(self, **kwargs):
        # we just pass this through
        return self.DVGeo.addRefAxis(**kwargs)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))

        if mode == 'rev' and ni > 0:
            for ptSetName in self.DVGeo.ptSetNames:
                dout = d_outputs[ptSetName].reshape(len(d_outputs[ptSetName])//3, 3)
                xdot = self.DVGeo.totalSensitivityTransProd(dout, ptSetName)

                # loop over dvs and accumulate
                xdotg = {}
                for k in xdot:
                    # check if this dv is present
                    if k in d_inputs:
                        # do the allreduce
                        # TODO reove the allreduce when this is fixed in openmdao
                        # reduce the result ourselves for now. ideally, openmdao will do the reduction itself when this is fixed. this is because the bcast is also done by openmdao (pyoptsparse, but regardless, it is not done here, so reduce should also not be done here)
                        xdotg[k] = self.comm.allreduce(xdot[k], op=MPI.SUM)

                        # accumulate in the dict
                        d_inputs[k] += xdotg[k]

# class that actually calls the dvgeometry methods
class DVGeoComp(om.ExplicitComponent):




    def initialize(self):

        self.options.declare('ffd_file', allow_none=False)
        self.options.declare('output_dir', default='./')
        self.initialized = False

        self.options['distributed'] = True

        # function used to add all the design variables, etc.
        self.options.declare('setup_dvgeo',recordable=False)
        self.options.declare('setup_dvcon', default=None, recordable=False)


    def setup(self):
        # create the DVGeo object that does the computations
        ffd_file = self.options['ffd_file']
        # self.DVGeo = DVGeometry(ffd_file)

        getDVGeo = self.options['setup_dvgeo']
        self.DVGeo = getDVGeo(ffd_file)
        self.add_input('pts', shape_by_conn=True)
    
        # iterate over the design variables for this comp
        
        dvs = self.DVGeo.getValues()
        
        for key, val in dvs.items():
            print('adding', key, val.size)
            self.add_input(key, src_indices=np.arange(val.size), val=val.real)
        
        self.add_output('deformed_pts', copy_shape='pts')

        self.count = 0
        # --- Setup the constraints ---
        if self.options['setup_dvcon']:
            getDVCon = self.options['setup_dvcon']

            self.DVCon = DVConstraints()
            self.DVCon.setDVGeo(self.DVGeo)

            # set a FAKE surface so the constraints will be initalized
            # triangulate the FFD to use as the fake surface

            # take the first vol for now, in the future this should loop over each and give them different names
            conn = self.DVGeo.FFD.topo.lIndex[0]

            def getTriMesh(face_conn):
                "returns the triagnulated mesh for given face points"

                face_p = [[], [], []]

                for i in range(face_conn.shape[0]-1):
                    for j in range(face_conn.shape[1]-1):
                        loc_pts = self.DVGeo.FFD.coef[face_conn[i:i+2, j:j+2]]

                        tri_1_pts = np.array([loc_pts[0, 0], loc_pts[1,0], loc_pts[0,1]])
                        tri_2_pts = np.array([loc_pts[1, 1], loc_pts[0,1], loc_pts[1,0]])


                        for p in range(len(face_p)):
                            face_p[p].append(tri_1_pts[p])
                            face_p[p].append(tri_2_pts[p])

                return face_p



            surf_p = [[], [], []]
            # iterate over highest and lowest index of each dimension
            for i in [0, -1]:

                for idx_dim in range(3):
                    s = slice(None)  # :

                    slicer = [s]*3  # [:, :, :]
                    slicer[idx_dim] = i   # [:, :, i]
                    nodes = conn[tuple(slicer)]
                    face_p = getTriMesh(nodes)

                    for p in range(len(face_p)):
                        surf_p[p].extend(face_p[p])

            p0 = np.array(surf_p[0])
            p1 = np.array(surf_p[1])
            p2 = np.array(surf_p[2])

            self.DVCon.setSurface(surf_p, format='point-point')

            #add all the constraints to the con object
            self.DVCon = getDVCon(self.DVCon)

            # loop over linear contraints
            for conName, con in self.DVCon.linearCon.items():

                print(f'for conName {conName}, adding {con.name} with size {con.ncon}')
                self.add_output(con.name, shape=con.ncon)

            for con_type, cons in self.DVCon.constraints.items():
                for conName, con in cons.items():


                    print(f'for conName {conName}, adding {con.name} with size {con.nCon}')
                    self.add_output(con.name, shape=con.nCon)


            
            self.add_input('tri_surf_p0', shape_by_conn=True)
            self.add_input('tri_surf_v1', shape_by_conn=True)
            self.add_input('tri_surf_v2', shape_by_conn=True)


    def compute(self, inputs, outputs):
        self.DVGeo.setDesignVars(inputs)

        if not self.initialized:
            self.initialized = True
            pts = inputs['pts']
            pts = pts.reshape(pts.size//3, 3)
            
 

            self.DVGeo.addPointSet(pts, 'pt_set', eps=1e-10, recompute=False)

            
            if MPI.COMM_WORLD.rank == 0:
                
                for folder in ['ffd', 'pts']:
                    path = os.path.join(self.options['output_dir'], folder)
                    os.system("mkdir -p %s" % (path))


            if self.options['setup_dvcon']:
                # initalize the constraint object too if we have one

                surf = [inputs['tri_surf_p0'], inputs['tri_surf_v1'], inputs['tri_surf_v2']]

                self.DVCon.setSurface(surf, format='point-point')

                self.DVCon = self.options['setup_dvcon'](self.DVCon)
        

        pts = self.DVGeo.update('pt_set')
        outputs['deformed_pts'] = pts.flatten()
        if self.options['setup_dvcon']:
            funcs = {}

            self.DVCon.evalFunctions(funcs, includeLinear=True)

            for func, val in funcs.items():
                outputs[func] = val


        # write the tecplot data out for viz
        self.DVGeo.writeTecplot(os.path.join(self.options['output_dir'], 'ffd/iter_{:03d}.dat'.format(self.count)), solutionTime=self.count)
        # self.DVGeo.writeTecplot('./output/ffd/iter_{:03d}.dat'.format(self.count), solutionTime=self.count)
        self.DVGeo.writePointSet('pt_set', os.path.join(self.options['output_dir'], 'pts/pts_{:03d}'.format(self.count)))
        
        if self.options['setup_dvcon']:
            self.DVCon.writeTecplot(os.path.join(self.options['output_dir'], "pts/cons_{:03d}.dat".format(self.count)), solutionTime=self.count)
                
        self.count += 1




    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))

        # import ipdb; ipdb.set_trace()
        if mode == 'rev' and ni > 0:
            # for ptSetName in self.DVGeo.ptSetNames:
            ptSetName = 'pt_set'
            dout = d_outputs['deformed_pts'].reshape(len(d_outputs['deformed_pts'])//3, 3)
            xdot = self.DVGeo.totalSensitivity(dout, ptSetName, comm=self.comm) # this has a all reduce inside
            # loop over dvs and accumulate
            
            xdotg = {}
            for k in xdot:
                # check if this dv is present
                if k in d_inputs:
                    d_inputs[k] += xdot[k].flatten()

            if self.options['setup_dvcon']:

                funcsSens = {}
                self.DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
                for constraintname in funcsSens:
                    for dvname in funcsSens[constraintname]:
                        if dvname in d_inputs:
                            dcdx = funcsSens[constraintname][dvname]
                            # if self.comm.rank == 0:
                            dout = d_outputs[constraintname]
                            jvtmp = np.dot(np.transpose(dcdx),dout)
                            d_inputs[dvname] += jvtmp
