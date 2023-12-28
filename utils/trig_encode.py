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