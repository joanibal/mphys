import openmdao.api as om
from pygeo import DVGeometry
from mpi4py import MPI
import numpy as np

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
        # self.options['distributed'] = True

        self.initialized = False

        self.options['distributed'] = True

        # function used to add all the design variables, etc.
        self.options.declare('setup_dvgeo',recordable=False)


    def setup(self):
        # create the DVGeo object that does the computations
        ffd_file = self.options['ffd_file']
        # self.DVGeo = DVGeometry(ffd_file)

        getDVGeo = self.options['setup_dvgeo']
        self.DVGeo = getDVGeo(ffd_file)
        self.add_input('pts', shape_by_conn=True)

        # iterate over the design variables for this comp
        varLists = {'globalVars':self.DVGeo.DV_listGlobal,
                   'localVars':self.DVGeo.DV_listLocal,
                   'sectionlocalVars':self.DVGeo.DV_listSectionLocal,
                   'spanwiselocalVars':self.DVGeo.DV_listSpanwiseLocal}
        for lst in varLists:
            for key in varLists[lst]:
                dv = varLists[lst][key]
                print(dv.value.shape)
                self.add_input(dv.name, src_indices=np.arange(dv.value.size), val=np.zeros(dv.value.shape))
        self.add_output('deformed_pts', copy_shape='pts')

    def compute(self, inputs, outputs):
        self.DVGeo.setDesignVars(inputs)

        if not self.initialized:
            pts = inputs['pts']
            self.DVGeo.addPointSet(pts.reshape(pts.size//3, 3), 'pt_set')

        # self.DVGeo.demoDesignVars('/home/josh/Dropbox/mphys_aerothermal/conjugate_hx_naca0012/output/', pointSet='pt_set')
        # self.DVGeo.checkDerivatives('pt_set')
        # quit()

        outputs['deformed_pts'] = self.DVGeo.update('pt_set').flatten()


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))
        print(list(d_inputs.keys()))

        # import ipdb; ipdb.set_trace()
        if mode == 'rev' and ni > 0:
            # for ptSetName in self.DVGeo.ptSetNames:
                ptSetName = 'pt_set'
                dout = d_outputs['deformed_pts'].reshape(len(d_outputs['deformed_pts'])//3, 3)
                xdot = self.DVGeo.totalSensitivityTransProd(dout, ptSetName)
                # loop over dvs and accumulate
                xdotg = {}
                for k in xdot:
                    # check if this dv is present
                    if k in d_inputs:
                        # do the allreduce
                        # TODO reove the allreduce when this is fixed in openmdao
                        # reduce the result ourselves for now. ideally, openmdao will do the reduction itself when this is fixed. this is because the bcast is also done by openmdao (pyoptsparse, but regardless, it is not done here, so reduce should also not be done here)
                        print(self.comm.rank, k , xdot)
                        xdotg[k] = self.comm.allreduce(xdot[k], op=MPI.SUM)

                        # accumulate in the dict
                        d_inputs[k] += xdotg[k]
