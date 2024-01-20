import numpy as np

def trig_encode(v, norm_v, trig_func="sin"):
    if trig_func == "sin":
        v_encoded = np.sin(2 * np.pi * v / norm_v)
    elif trig_func == "cos":
        v_encoded = np.cos(2 * np.pi * v / norm_v)
    else:
        print("Function not implemented")
        return None
    return v_encoded

def trig_decode(vsin, vcos, norm_v):
    varcsin = np.arcsin(vsin)
    if varcsin < 0:
        va = np.array([np.pi - varcsin, 2*np.pi + varcsin])
    else:
        va = np.array([varcsin, np.pi - varcsin])
    varccos = np.arccos(vcos)
    vb = np.array([varccos, 2*np.pi - varccos])
    va = np.round(va, 3)
    vb = np.round(vb, 3)
    v = np.intersect1d(va, vb)[0]


    return int(np.round(v * norm_v / (2 * np.pi), 0))