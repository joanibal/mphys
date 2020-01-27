#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np

from openmdao.api import Group, ExplicitComponent
from omfsi.assembler import OmfsiAssembler

class FsiAssembler(object):
    def __init__(self,struct_assembler,aero_assembler,xfer_assembler,geodisp_assembler=None):
        """
        Fluid structure interaction problem assembler.
        Given :mod:`OmfsiAssemblers` for
        the structural solver, the aerodynamic solver, and the transfer scheme
        (load and displacement transfer modules), the FsiAssembler wraps the
        subsystem assembler calls so that adding the fsi modules to a model
        requires only two calls


        Parameters
        ----------
        struct_assembler : :mod:`OmfsiAssembler`
            The structural solver assembler
        aero_assembler : :mod:`OmfsiAssembler`
            The aerodynamic solver assembler
        xfer_assembler : :mod:`OmfsiAssembler`
            The load and displacement transfer scheme assembler
        geodisp_assembler : :mod:`OmfsiAssembler`
            The geometry displacement addition assembler. This module adds
            aeroelastic surface displaces to the geometry-modified aerodynamic
            surface. If a geodisp_assembler is not provided, the default
            assembler and geodisp component will be used.
        """
        self.struct_assembler = struct_assembler
        self.aero_assembler   = aero_assembler
        self.xfer_assembler   = xfer_assembler
        if geodisp_assembler is None:
            self.geodisp_assembler = GeoDispAssembler(aero_assembler)
        self.connection_srcs = {}

    def add_model_components(self,model):
        """
        Add model-level FSI subsystem modules.

        Parameters
        ----------
        model : openmdao model
            The model to which the fsi modules will be added
        """
        self.geodisp_assembler.add_model_components(model,self.connection_srcs)
        self.xfer_assembler.add_model_components(model,self.connection_srcs)
        self.struct_assembler.add_model_components(model,self.connection_srcs)
        self.aero_assembler.add_model_components(model,self.connection_srcs)

    def add_fsi_subsystem(self,model,scenario):
        """
        Add FSI subsystem modules to a scenario in three phases.
          | 1) Add the FSI group and fsi level subsystem components
          | 2) Add the scenario level components
          | 3) Form the input connections for the added components

        Parameters
        ----------
        model : openmdao model
            The openmdao model which
        scenario : openmdao group
            The scenario to which the fsi modules will be added
        """
        fsi_group = scenario.add_subsystem('fsi_group',Group())

        self.geodisp_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)
        self.xfer_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)
        self.struct_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)
        self.aero_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)

        self.geodisp_assembler.add_scenario_components(model,scenario,self.connection_srcs)
        self.xfer_assembler.add_scenario_components(model,scenario,self.connection_srcs)
        self.struct_assembler.add_scenario_components(model,scenario,self.connection_srcs)
        self.aero_assembler.add_scenario_components(model,scenario,self.connection_srcs)

        self.geodisp_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)
        self.xfer_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)
        self.struct_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)
        self.aero_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)

        self._reorder_fsi_group(fsi_group)

        return fsi_group

    def _reorder_fsi_group(self,fsi_group):

        # pull out the component names
        comp_list = []
        for comp in fsi_group._static_subsystems_allprocs:
            comp_list.append(comp.name)

        # set the order of known fsi components if they exist
        new_list = ['disp_xfer','geo_disp','aero','load_xfer','struct']
        new_order = []
        for comp_name in new_list:
            if comp_name in comp_list:
                new_order.append(comp_list.pop(comp_list.index(comp_name)))

        # append any other components in the group
        new_order.extend(comp_list)

        fsi_group.set_order(new_order)


class GeoDispAssembler(OmfsiAssembler):
    def __init__(self,aero_assembler):
        self.aero_assembler = aero_assembler
    def get_aero_nnodes(self):
        return self.aero_assembler.get_nnodes()

    def add_model_components(self,model,connection_srcs):
        pass
    def add_scenario_components(self,model,scenario,connection_srcs):
        pass
    def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):
        fsi_group.add_subsystem('geo_disp',GeoDisp(get_aero_nnodes=self.get_aero_nnodes))
        connection_srcs['x_a'] = scenario.name+'.'+fsi_group.name+'.geo_disp.x_a'
    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        model.connect(connection_srcs['x_a0'],scenario.name+'.'+fsi_group.name+'.geo_disp.x_a0')
        model.connect(connection_srcs['u_a'],scenario.name+'.'+fsi_group.name+'.geo_disp.u_a')

class GeoDisp(ExplicitComponent):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options.declare('get_aero_nnodes',default=None,desc='function to get number of aerodynamic nodes')
        self.options['distributed'] = True

    def setup(self):
        local_size = self.options['get_aero_nnodes']() * 3
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a0',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')
        self.add_input('u_a', shape=local_size,val=np.zeros(local_size),src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface displacements')

        self.add_output('x_a',shape=local_size,desc='deformed aerodynamic surface')

    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_outputs['x_a'] += d_inputs['x_a0']
                if 'u_a' in d_inputs:
                    d_outputs['x_a'] += d_inputs['u_a']
        if mode == 'rev':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_inputs['x_a0'] += d_outputs['x_a']
                if 'u_a' in d_inputs:
                    d_inputs['u_a']  += d_outputs['x_a']
