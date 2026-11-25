import numpy as np, pywt
s,re,im=np.load('MT_haar.npy'); MT=re+1j*im
ex=np.pi/(8*np.log(2)); print("pi/(8ln2)=",ex," c0=",-1/ex)
for Q in [1,2,4,5,8]:
    MQ=np.array([np.mean([MT[(i+q*40//Q)%40] for q in range(Q)]) for i in range(40)])
    print(f"Q={Q}: Im range [{MQ.imag.min():.4f},{MQ.imag.max():.4f}]  max rel dev from pi/(8ln2) = {np.max(np.abs(MQ.imag-ex))/ex:.2e}")
# band truncation of H's wavelet matrix (single fixed lattice)
N=1024; L=7
k=np.fft.fftfreq(N)*N; mH=-1j*np.sign(k)
HM=np.real(np.fft.ifft(mH[:,None]*np.fft.fft(np.eye(N),axis=0),axis=0))
res={}
for w in ['haar','db2','db4']:
    W=np.array([np.concatenate(pywt.wavedec(e,w,mode='periodization',level=L)) for e in np.eye(N)]).T
    M=W@HM@W.T
    lev=[L]*(N>>L); kk=list(range(N>>L)); isA=np.ones(N>>L,bool).tolist()
    for l in range(L,0,-1):
        n=N>>l; lev+=[l]*n; kk+=list(range(n)); isA+=[False]*n
    lev=np.array(lev); kk=np.array(kk); isA=np.array(isA)
    cen=(kk+0.5)*2.0**lev      # cell centre in samples
    dl=np.abs(lev[:,None]-lev[None,:]); big=np.maximum(2.0**lev[:,None],2.0**lev[None,:])
    d=np.abs(cen[:,None]-cen[None,:]); d=np.minimum(d,N-d)/big   # periodic distance in units of larger cell
    out=[]
    for R in [1,2,4,8,16]:
        mask=(dl<=1)&(d<=R)
        out.append(np.linalg.norm(M-M*mask,2))
    res[w]=out; print(w,'||H - band_R(H)||, R=1,2,4,8,16 (|dlevel|<=1):',["%.2e"%v for v in out])
    Rs=np.array([4,8,16]); print('   log2 slope',round(np.polyfit(np.log2(Rs),np.log2(out[2:]),1)[0],2))
np.save('e6.npy',res,allow_pickle=True)
