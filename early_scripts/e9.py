import numpy as np
from e8 import m0
u=np.linspace(-10,6,1600); t=2.0**u
out={}
for name,m in [('haar',0),('db2',1),('db4',4),('sym4',3),('coif3',8),('db4',0),('db2',0),('sym4',0),('coif3',0)]:
    k=np.trapezoid(m0(name,m,t),u)
    s=np.linspace(0,1,41)[:-1]; js=np.arange(-12,9)
    MT=np.array([np.sum(m0(name,m,2.0**(ss+js))) for ss in s])
    row=[]
    for Q in [1,2,4,5]:
        MQ=np.array([np.mean([MT[(i+q*40//Q)%40] for q in range(Q)]) for i in range(40)])
        row.append(np.max(np.abs(MQ-k))/abs(k.imag))
    out[(name,m)]=(k,row,MT)
    print(f"{name} m={m}: kappa={k.real:+.3f}{k.imag:+.3f}i |c0|={1/abs(k.imag):.2f} var x{(1/abs(k.imag))**2/(8*np.log(2)/np.pi)**2:.1f} ripple Q=1,2,4,5: {[f'{r:.1%}' for r in row]}")
np.save('e9.npy',{str(k):(v[0],v[1],v[2]) for k,v in out.items()},allow_pickle=True)
g=np.linspace(-2.5,2.5,201); tt=2.0**g
np.save('e9_int.npy',np.vstack([g,m0('haar',0,tt).imag,m0('db4',0,tt).imag,m0('db4',4,tt).imag]))
