import numpy as np
import string
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--file", help="FFD file ",
                    type=str, default='None')
parser.add_argument("--angle", help="distance to move x coord",
                    type=str, default='0')
parser.add_argument("--point", help="point to rotate about",
                    type=str, default='0')
parser.add_argument("--output", help="name of output file",
                    type=str, default='0')


args = parser.parse_args()



if args.file=='None':
    raise IOError('no input file was specified') 



f= open(args.file,'r')
flines = f.readlines()

[num_X, num_Y, num_Z] = [int(i) for i in string.split(flines[1])]


x = np.asarray([float(i) for i in string.split(flines[3])] )
for j in range(4,3+(num_Y*num_Z)):
     x = np.vstack((x, [float(i) for i in string.split(flines[j])] ))

y = np.asarray([float(i) for i in string.split(flines[j+2])] )
for j in range(j+3,j+2+(num_Y*num_Z)):
     y = np.vstack((y, [float(i) for i in string.split(flines[j])] ))

z = np.asarray([float(i) for i in string.split(flines[j+2])] )
for j in range(j+3,j+2+(num_Y*num_Z)):
     z = np.vstack((z, [float(i) for i in string.split(flines[j])] ))

f.close()
 # Define 2D postion vector

x_avg = sum(sum(x)/4)/(len(x[0]))
y_avg = sum(sum(y)/4)/(len(y[0]))

# print x_avg, y_avg

R = np.sqrt((x-x_avg)**2 + (y-y_avg)**2)
ang = np.arctan2((y-y_avg),(x-x_avg)) + float(args.angle)*np.pi/180

x_new = np.cos(ang)*R + x_avg
y_new = np.sin(ang)*R + y_avg

f = open(args.output, 'w')
f.write('1 \n')
f.write(str(num_X) + ' ' + str(num_Y) + ' ' + str(num_Z) + '\n \n')



for j in range(0,(num_Y*num_Z)):
     f.write(' '.join(map(str, x_new[j,:])) + '\n')
f.write('\n')

for j in range(0,(num_Y*num_Z)):
     f.write(' '.join(map(str, y_new[j,:])) + '\n')
f.write('\n')

for j in range(0,(num_Y*num_Z)):
     f.write(' '.join(map(str, z[j,:])) + '\n')
f.write('\n')




f.close()





# x =[]
# y=[]
# for line in f:

# 	x.append(float(string.split(line)[0]))
# 	y.append(float(string.split(line)[1]))

# x = np.asarray(x)
# y = np.asarray(y)

# x = x + float(args.distx);
# y = y + float(args.disty);

# print(min(x))

# f = open(args.file, 'w')
# # f.write('opt airfoil\n')

# for i in range(0, len(x)):
# 	# print(i)
# 	f.write(str(x[i]) + ' ' + str(y[i]) + '\n')

# f.close()