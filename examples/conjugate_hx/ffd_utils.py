""" holds a few useful functions for working with FFDs"""
import string
import numpy as np


def readFFDFile(filename):

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    f = open(filename, 'r')

    data = f.read()
    data = data.split()
    nVol = int(data[0])
    nxPts = int(data[1])
    nyPts = int(data[2])
    nzPts = int(data[3])
    
    nPts = nxPts*nyPts*nzPts
    coords = np.zeros((nPts, 3))*np.NAN

    jj = 4
    for ii in range(3*nPts):
    #fill in the x coordinates
        idx = np.mod(ii, nPts)
        dim = ii//nPts 

        if is_number(data[jj]):
            coords[idx, dim] = float(data[jj])
        
        jj +=1
    
    if  (coords == np.NAN).any():
        print('there was not enough data for the amount of pts specified')
        raise ValueError
    elif (jj < nPts):
        print('WARNING: not all the data in the file was used')
        
   


    return coords, (nxPts, nyPts, nzPts)


def writeFFDFile(filename, coords, ffd_size):



    f = open(filename, 'w')

    f.write('1 \n') # dosn't support multi volume FFDs
    f.write('{:<5} {:<5} {:<5}'.format(*ffd_size))
    f.write('\n\n\n')

    nPts = np.prod(ffd_size)
    # coords = np.zeros((nPts, 3))*np.NAN

    nxPts,nyPts, nzPts = ffd_size
    

    jj = 4
    for ii in range(3*nPts):
    #fill in the x coordinates
        idx = np.mod(ii, nPts)
        dim = ii//nPts 

        f.write('{:>12.6f}'.format(coords[idx,dim]))

        if (np.mod(ii+1,ffd_size[0]) == 0):
            f.write('\n')
        
        if (np.mod(ii+1,ffd_size[0]*ffd_size[1]) == 0):
            f.write('\n')
        
        if (idx+1 == nPts):
            f.write('\n')
        
        jj +=1
    
    
    f.close()


    return coords, (nxPts, nyPts, nzPts)



def getSections(coords, ffd_size, section_idx=0):
    """
        split the list of the FFD point coordinates into subsets of 
        sections along the given section idx
        
    Parameters:
        coords: np.array(nPts, 3)
            list of the coordinates which define the ffd.
            *IT IS ASSUMED TO BE ORDERED ACCORDING TO readFFD()*
            that is all x point, then y, then z
        
        ffd_size: tuple(3)
            tuple of points along the x, y, and z axis 
        
        section_idx: int
            idx used to split the ffd up into sections 

    Outputs:
        sections: np.array(nSections, nSecPts, nDim)
            coordinates of the points that make up each section
    """
    if section_idx== 0:
        sections = np.zeros((ffd_size[0], ffd_size[1]*ffd_size[2], 3))

        for idx_slice in range(ffd_size[0]):
            sections[idx_slice] = coords[idx_slice::ffd_size[0], :]

    else:
        raise NotImplementedError


    return sections


if __name__ == '__main__':
   coords, ffd_size =  readFFDFile('ffds/nacelle_15_8.xyz')

   coords[:, 1:] *= 1.2

   writeFFDFile('test.xyz', coords, ffd_size)