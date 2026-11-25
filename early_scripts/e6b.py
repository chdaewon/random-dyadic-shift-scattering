import numpy as np, pywt
N=1024; L=7
k=np.fft.fftfreq(N)*N; mH=-1j*np.sign(k); mH[N//2]=0
HM=np.real(np.fft.ifft(mH[:,None]*np.fft.fft(np.eye(N),axis=0),axis=0))
for w in ['haar','db4']:
    W=np.array([np.concatenate(pywt.wavedec(e,w,mode='periodization',level=L)) for e in np.eye(N)]).T
    M=W@HM@W.T
    lev=np.array([L]*(N>>L)+sum([[l]*(N>>l) for l in range(L,0,-1)],[]))
    dl=np.abs(lev[:,None]-lev[None,:])
    for q in [0,1,2,3]:
        print(w,'q=',q,'||M - M[dl<=q]||=',round(np.linalg.norm(M-M*(dl<=q),2),3), ' HS frac', round(np.linalg.norm(M-M*(dl<=q))/np.linalg.norm(M),4))
