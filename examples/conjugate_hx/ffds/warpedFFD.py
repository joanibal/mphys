import os
import shutil
import numpy as np
# note, need to add a single point so there is something to project to and call the update
DVGeo.addPointSet(np.array([[0,0,0]]), "foo")
# DVCon.writeTecplot('constraint_pts_check.dat')
#print(DVGeo.getVarNames())
counter = 0
for i_var in range(3):

x_dv = {
    
     'nacelle_geom': [0.,0.,0.],

 'shape': np.zeros(len(index_list))

}

n_steps = 15

for i_transition, val in enumerate(np.linspace(0,.5, n_steps)):
x_dv['nacelle_geom'][i_var] = val
DVGeo.setDesignVars(x_dv)
DVGeo.update("foo")
DVGeo.writeTecplot('geo_test.dat')
os.system('tec360 -b ffd_check.lay -p ffd_movie.mcr')
file_name = 'movie_images/ffd_movie_{:05d}.png'.format(n_steps*i_var+i_transition)
shutil.move('ffd_movie.png', file_name)
# print("foobar", file_name)
counter += 1
# need to keep the index of the loop over the global variables so the image file names stay ordered properly
i_nac_geom = counter
for i_var in range(len(index_list)):

x_dv = {
    
     'nacelle_geom': [0.,0.,0.],

 'shape': np.zeros(len(index_list))

}

n_steps = 15

for i_transition, val in enumerate(np.linspace(0,.5, n_steps)):
x_dv['shape'][i_var] = val
DVGeo.setDesignVars(x_dv)
DVGeo.update("foo")
DVGeo.writeTecplot('geo_test.dat')
os.system('tec360 -b ffd_check.lay -p ffd_movie.mcr')
file_name = 'movie_images/ffd_movie_{:05d}.png'.format(i_nac_geom + n_steps*i_var+i_transition)
shutil.move('ffd_movie.png', file_name)
print("foobar", file_name)