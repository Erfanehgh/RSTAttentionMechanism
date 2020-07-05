import numpy as np

def Write_Weights(W1, W2, WSat, WNu, W1_File, W2_File, WSat_File, WNu_File):
    np.savetxt(W1_File, W1)
    np.savetxt(W2_File, W2)
    np.savetxt(WSat_File, WSat)
    np.savetxt(WNu_File, WNu)#, fmt='%.2f')