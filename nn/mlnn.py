"""
multilayer nn
"""

import numpy as np

def build_net(net_desc):
    n = len(net_desc)
    layers = n-1
    w = []
    for l in range(layers):
        # wsize = sizeout x sizein
        w += [ np.zeros((net_desc[l+1],net_desc[l])) ]
    return w

def forward(X,W,f):
    """
        X input
        W weights
        f activation functio
    """
    return f(np.dot(X,W.T))

def main():
    f = np.sign
    net_desc = [3,2,1] # describe the net
    w = build_net(net_desc)
    x = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    x1= forward(x,w[0],f)
    x2 = forward(x1,w[1],f)
    print x,x1,x2

if __name__ == "__main__":
    main()
