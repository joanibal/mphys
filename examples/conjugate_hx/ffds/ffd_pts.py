# python script to determine the y and z points for the ffd
import numpy as np
import matplotlib.pyplot as plt
n = 8

ang = np.pi*2 / n
print ang
R = 0.1


pt = np.array


L = np.array([[np.cos(ang/2), - np.sin(ang/2)],
              [np.sin(ang/2),   np.cos(ang/2)]])
pts = np.array([np.matmul(L, np.array([0, -R]))])


L = np.array([[np.cos(ang), - np.sin(ang)],
              [np.sin(ang),   np.cos(ang)]])


for i in range(1, n):
    pt = np.matmul(L, pts[-1])

    pts = np.vstack((pts, pt))


print('----- z -------')

print pts[:, 0]*0.2
print pts[:, 0]*0.4
print pts[:, 0]*0.6
print pts[:, 0]*0.8

print('----- -------')


print pts[:, 0]

print('----- -------')


print pts[:, 0]*((1-0.0029/0.08)/0.5) * 0.1
print pts[:, 0]*((1-0.0029/0.08)/0.5) * 0.2
print pts[:, 0]*((1-0.0029/0.08)/0.5) * 0.3
print pts[:, 0]*((1-0.0029/0.08)/0.5) * 0.4
print pts[:, 0]*((1-0.0029/0.08)/0.5) * 0.5


print('----- y -------')
print pts[:, 1]*0.2
print pts[:, 1]*0.4
print pts[:, 1]*0.6
print pts[:, 1]*0.8

print('----- -------')

print pts[:, 1]
print('----- -------')

print pts[:, 1]*((1-0.0029/0.08)/0.5) * 0.1 + np.sin(0.1528)*0.1
print pts[:, 1]*((1-0.0029/0.08)/0.5) * 0.2 + np.sin(0.1528)*0.2
print pts[:, 1]*((1-0.0029/0.08)/0.5) * 0.3 + np.sin(0.1528)*0.3
print pts[:, 1]*((1-0.0029/0.08)/0.5) * 0.4 + np.sin(0.1528)*0.4
print pts[:, 1]*((1-0.0029/0.08)/0.5) * 0.5 + np.sin(0.1528)*0.5


# plt.plot(pts[:, 0], pts[:, 1])
# plt.show()
