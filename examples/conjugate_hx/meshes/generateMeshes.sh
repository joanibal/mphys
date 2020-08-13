
# cp meshes/visc_wall_xrefine/plate_1e-08.cgns meshes/thermal_visc_wall_xrefine

# cgns_utils removebc 2D/nacelle_2D.cgns 2D/nacelle_2D_rot.cgns
# cgns_utils rotate  2D/nacelle_2D_rot.cgns 0 1 0 90 2D/nacelle_2D_rot.cgns
# cgns_utils scale 2D/nacelle_2D_rot.cgns 0.0254 2D/nacelle_2D_rot.cgns
# python runNacelle.py --file=2D/nacelle_2D_rot.cgns --out=2D/nacelle_2D_vol.cgns --s0=1e-5


# cgns_utils removebc axisymmetric/nacelle_axisym.cgns axisymmetric/test2.cgns
# cgns_utils rotate axisymmetric/test2.cgns 0 1 0 90 axisymmetric/test2.cgns
# cgns_utils scale axisymmetric/test2.cgns 0.0254 axisymmetric/test2.cgns
# python runNacelle.py --file=axisymmetric/test2.cgns --out=axisymmetric/test2_vol.cgns --s0=1e-5


# cgns_utils scale nacelle_coarse_rotated.cgns 0.0254 nacelle_coarse_rotated.cgns
# python runNacelle.py --file=nacelle_coarse_rotated.cgns --out=nacelle_coarse_vol.cgns --s0=1e-5
# cgns_utils removebc nacelle_coarse_vol.cgns
# cgns_utils overwritebc nacelle_coarse_vol.cgns thermal_visc_boco.info

# cgns_utils rotate nacelle_fine.cgns 0 1 0 90 nacelle_fine_rotated.cgns
# python runNacelle.py --file=nacelle_fine_rotated.cgns --out=nacelle_fine_vol.cgns --s0=1e-5
# cgns_utils removebc nacelle_fine_vol.cgns
# cgns_utils overwritebc nacelle_fine_vol.cgns thermal_visc_boco.info

# cgns_utils rotate nacelle_finest.cgns 0 1 0 90 nacelle_finest_rotated.cgns
# cgns_utils scale nacelle_finest_rotated.cgns 0.0254 nacelle_finest_rotated.cgns
# python runNacelle.py --file=nacelle_finest_rotated.cgns --out=nacelle_finest_vol.cgns --s0=1e-5
# cgns_utils removebc nacelle_finest_vol.cgns
# cgns_utils overwritebc nacelle_finest_vol.cgns thermal_visc_boco.info


# axisymmetric meshes
cgns_utils rotate axisymmetric/nacelle_axisym_vol.cgns 0 1 0 90 axisymmetric/nacelle_axisym_vol.cgns
cgns_utils scale axisymmetric/nacelle_axisym_vol.cgns 0.0254 axisymmetric/nacelle_axisym_vol.cgns
# python runNacelle.py --file=axisymmetric/test2.cgns --out=axisymmetric/test2_vol.cgns --s0=1e-5

