import openmdao.api as om
from pygeo import DVGeometry, DVConstraints
from mpi4py import MPI
import numpy as np
from openmdao.utils.array_utils import evenly_distrib_idxs
import time

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('ffd_file', allow_none=False)
        self.options['distributed'] = True

    def setup(self):
        # create the DVGeo object that does the computations
        ffd_file = self.options['ffd_file']
        self.DVGeo = DVGeometry(ffd_file)
        self.DVCon = DVConstraints()
        self.DVCon.setDVGeo(self.DVGeo)
        self.omPtSetList = []

    def compute(self, inputs, outputs):
        # inputs are the geometric design variables
        self.DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptName in self.DVGeo.points:
            if ptName in self.omPtSetList:
                # update this pointset and write it as output
                outputs[ptName] = self.DVGeo.update(ptName).flatten()

        # compute the DVCon constraint values
        constraintfunc = dict()
        self.DVCon.evalFunctions(constraintfunc, includeLinear=True)
        comm = self.comm
        if comm.rank == 0:
            for constraintname in constraintfunc:
                outputs[constraintname] = constraintfunc[constraintname]

    def nom_addPointSet(self, points, ptName, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points)//3, 3), ptName, **kwargs)
        self.omPtSetList.append(ptName)

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

    def nom_addGeoDVLocal(self, dvName, axis='y'):
        nVal = self.DVGeo.addGeoDVLocal(dvName, axis=axis)
        self.add_input(dvName, shape=nVal)
        return nVal

    def nom_addThicknessConstraints2D(self, name, leList, teList, nSpan=10, nChord=10):
        self.DVCon.addThicknessConstraints2D(leList, teList, nSpan, nChord, lower=1.0, name=name)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, val=np.ones((nSpan*nChord,)), shape=nSpan*nChord)
        else:
            self.add_output(name, shape=(0,))


    def nom_addVolumeConstraint(self, name, leList, teList, nSpan=10, nChord=10):
        self.DVCon.addVolumeConstraint(leList, teList, nSpan=nSpan, nChord=nChord, name=name)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, val=1.0)
        else:
            self.add_output(name, shape=0)

    def nom_add_LETEConstraint(self, name, volID, faceID):
        self.DVCon.addLeTeConstraints(volID, faceID, name=name)
        # how many are there?
        conobj = self.DVCon.linearCon[name]
        nCon = len(conobj.indSetA)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, val=np.zeros((nCon,)), shape=nCon)
        else:
            self.add_output(name, shape=0)
        return nCon


    def nom_addRefAxis(self, **kwargs):
        # we just pass this through
        return self.DVGeo.addRefAxis(**kwargs)

    def nom_setConstraintSurface(self, surface):
        # constraint needs a triangulated reference surface at initialization
        self.DVCon.setSurface(surface)

    

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))
        
        if mode == 'rev' and ni > 0:
            constraintfuncsens = dict()
            self.DVCon.evalFunctionsSens(constraintfuncsens, includeLinear=True)
            for constraintname in constraintfuncsens:
                for dvname in constraintfuncsens[constraintname]:
                    if dvname in d_inputs:
                        dcdx = constraintfuncsens[constraintname][dvname]
                        if self.comm.rank == 0:
                            dout = d_outputs[constraintname]
                            jvtmp = np.dot(np.transpose(dcdx),dout)
                        else:
                            jvtmp = 0.0
                        d_inputs[dvname] += jvtmp
                        # OM does the reduction itself
                        # d_inputs[dvname] += self.comm.reduce(jvtmp, op=MPI.SUM, root=0)
                
            for ptSetName in self.DVGeo.ptSetNames:
                if ptSetName in self.omPtSetList:
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
