from matplotlib import pyplot as plt
import numpy as np

img = "/home/rishabh/work/siemens_work/60363_NormalizedOutput.raw"

A = np.fromfile(img, dtype='uint16', sep="")
print (A.shape)
B = A[36:]
print (B.shape)
B = np.reshape(B, (4, 167, 256, 256))
print (B.shape)
dose = B[3,61,:,:]
print (dose)
# dose = np.transpose(dose, (1,2,0))

print (dose.shape)
plt.imshow(dose)
plt.show()