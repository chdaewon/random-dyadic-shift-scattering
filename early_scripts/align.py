import numpy as np
from mult import psi_samples, ft
u=np.linspace(-10,6,1600); t=2.0**u
for name in ['haar','db2','db4','sym4','coif3']:
    x,p,dx=psi_samples(name,10)
    cen=np.sum(x*p**2)*dx
    ph=ft(x,p,dx,t); p2=ft(x,p,dx,t/2)
    row=[]
    for m in range(0,8):
        # Theta_m(s)=psi(2s-m)-psi(2s-m-1):  FT = 1/2 psi^(t/2) e^{-pi i m t}(1-e^{-pi i t})
        Th=0.5*p2*np.exp(-1j*np.pi*m*t)*(1-np.exp(-1j*np.pi*t))
        k=np.trapezoid(np.conj(ph)*Th,u)
        row.append(k)
    best=int(np.argmax([abs(k.imag) for k in row]))
    print(f"{name:5s} centroid={cen:.2f}  m=0: {row[0]:.3f}  best m={best}: {row[best]:.3f}  |Im|/|Re|={abs(row[best].imag)/max(abs(row[best].real),1e-9):.1f}")
