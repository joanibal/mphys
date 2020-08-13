from pyhyp import pyHyp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="surface mesh file ",
                    type=str, default='None')
parser.add_argument('--s0', help='inital of wall spacing', type=float, default=1e-5)
parser.add_argument("--out", help="output file",
                    type=str, default='None')
args = parser.parse_args()



if args.out=='None':
    args.out = args.file[:(len(args.file)-5)] + '_vol.cgns'

print(args.out)
options = {
    # ---------------------------
    #   General options
    # ---------------------------
    'inputFile': args.file,
    'fileType': 'cgns',
    'unattachedEdgesAreSymmetry': True,
    'outerFaceBC': 'farfield',
    'autoConnect': 'True',
    'BC': {},
    'families': 'wall',
    # ---------------------------
    #   Grid Parameters
    # ---------------------------
    'N': 65,
    's0': args.s0,
    'marchDist': 25,
    # ---------------------------
    #   Pseudo Grid Parameters
    # ---------------------------
    'ps0': -1,
    'pGridRatio': -1,
    'cMax': 1.0,


    # ---------------------------
    #   Smoothing parameters
    # ---------------------------
    'epsE': 1.0,
    'epsI': 2.0,
    'theta': 3.0,
    'volCoef': .2,
    'volBlend': 0.0005,
    'volSmoothIter': 20,

    # ---------------------------
    #   Solution Parameters
    # ---------------------------
    'kspRelTol': 1e-10,
    'kspMaxIts': 1500,
    'kspSubspaceSize': 50,


    'unattachedEdgesAreSymmetry': False,
    'autoConnect': True,
    'BC': {1: {'jLow': 'zSymm',
               'kHigh': 'zSymm'}},

}


hyp = pyHyp(options=options)
hyp.run()
hyp.writeCGNS(args.out)
