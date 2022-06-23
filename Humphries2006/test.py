import brainpy as bp
import brainpy.math as bm

a = bm.ones(5)

b = a.astype(bm.bool_)

print(a,b)