import numpy as np
import string
import argparse
import matplotlib.pyplot as plt
from pygeo.airfoil_utils import convert, readTecplotFEdata, genAirfoilCoords


parser = argparse.ArgumentParser()
parser.add_argument("--file", help="slice file ",
                    type=str, default='None')
parser.add_argument("--offset", help="offset from airfoil ",
                    type=float, default=0.75e-2)
parser.add_argument("--offsetAxisIndx", help="axis to apply offset x=0, y=1, z=2",
                    type=int, default=1)
parser.add_argument("--xOffset", help="offset of LE and TE",type=float, default=1e-4)

parser.add_argument("--nSections", help="number of chordwise sections",
                    type=int, default=5)
parser.add_argument("--output", help="name of output FFD file",
                    type=str, default='FFD.xyz')
parser.add_argument('--fixLE', help='useful with few sections', action='store_true')


# parser.add_argument("--x_sections", help="additional x location of FFD points",
                    # type=str, default=[])


args = parser.parse_args()



nPts = args.nSections * 2

# _, readTecplotFEdata('cruise_slice.dat')
data, conn = readTecplotFEdata('cruise_slice.dat')[1:]
data = data[:3]
nSlices = len(data)

def removeTE(sliceData, tol=1e-2):
    """ assumes that points are ordered clockwise starting at the top of the TE """
    newSliceData = sliceData
    for ii in range(len(sliceData)-1):
        delta = sliceData[ii+1, :3] -  sliceData[ii, :3]
        magdelta = np.linalg.norm(delta)

        # print 'asdf'
        # true when there is a relatively small change in x and the change in y is positive
        if (np.abs(delta[0]/magdelta) <  tol and delta[args.offsetAxisIndx] > 0.0):
            ii += 1
            # print 'tol', np.abs(delta[0]/magdelta)
            break

    return newSliceData[:ii]


newData = []
for ii in range(nSlices):
    newData.append(removeTE(convert(data[ii], conn[ii])))

    # there should be a step in here to de rotate the coords, then make the FFD and then rerotate the FFD

    coords, airfoilCurve = genAirfoilCoords(newData[ii][:,:3], nPts, findLE=True, repeatLE=True, repeatTE=True, equalPts=True, order=3)


    # # picks a new leading edge pt by doing a extending the pt tanget to the curve at the previous pt
    # # to the leading edge (x = 0)


    if args.fixLE:
        for sign in [-1, 1]:
            coords[nPts/2::sign, args.offsetAxisIndx] -= sign*args.offset

            s, pt = airfoilCurve.projectPoint(coords[nPts/2 + (sign - 1)/2 + sign, :])
            dX_ds = airfoilCurve.getDerivative(s)
            dy_dx = dX_ds[0]**-1*dX_ds[1]


            coords[nPts/2 + (sign - 1)/2, args.offsetAxisIndx] = -coords[nPts/2 + (sign - 1)/2 + sign, 0]*dy_dx +  coords[nPts/2 + (sign - 1)/2 + sign, 1]
    else:
        coords[nPts/2:, args.offsetAxisIndx] -= args.offset
        coords[:nPts/2, args.offsetAxisIndx] += args.offset


    coords[[nPts/2-1,nPts/2], 0] -= args.xOffset
    coords[[0, -1], 0] += args.xOffset


    plt.plot(coords[:,0], coords[:,1], '--o')
    plt.plot(newData[ii][:,0], newData[ii][:,1], '-o')
    plt.show()

    newData[ii] = coords






# -------------- write out the FFD file



num_X = args.nSections




# print newData
with open(args.output, 'w') as fid:

    # import ipdb; ipdb.set_trace()
    fid.write('1 \n')
    if args.offsetAxisIndx == 1:
        fid.write(str(num_X) + ' ' + str(2) + ' ' + str(nSlices) + '\n \n \n')
    elif args.offsetAxisIndx == 2:
        fid.write(str(num_X) + ' ' + str(nSlices) + ' ' + str(2) + '\n \n \n')


    for axis in range(3): # loop over each diminsion
        for ii in range(nSlices):
            for jj in range(2):  # always [0,1] for the top and bottom points (no points in the middle)
                start = -jj
                end = nPts/2 - jj
                inc = 1 - 2*jj

                fid.write(np.array2string(newData[ii][start:end:inc,axis], formatter={'float_kind':lambda x: "%.2f" % x})[1:-1])

                fid.write('\n')

            fid.write('\n')

        fid.write('\n\n')

