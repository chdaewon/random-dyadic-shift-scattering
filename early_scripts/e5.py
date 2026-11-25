import numpy as np, pywt
# (b) truncation bias ||P_{>J} f|| for smooth f, Haar vs db2 vs db4
N=2**14; x=np.arange(N)/N
f=np.exp(-((x-0.5)/0.08)**2)*np.cos(12*np.pi*x)   # smooth, ~periodic
f-=f.mean()
print("bias ||P_{>J}f||/||f|| ; J counts levels kept from coarse (cell=1/2^j)")
B={}
for w in ['haar','db2','db4']:
    c=pywt.wavedec(f,w,mode='periodization',level=10)  # details levels 1..10 ; level l cell = 2^l/N
    # energy at levels finer than cell 2^{-j}: levels l with 2^l/N < 2^{-j}  -> l < 14-j
    out=[]
    for j in range(3,10):
        e=sum(np.sum(c[10-l+1]**2) for l in range(1,14-j))   # levels 1..(13-j)
        out.append(np.sqrt(e)/np.linalg.norm(f))
    B[w]=out; print(w,["%.1e"%v for v in out])
    js=np.arange(3,10); sl=np.polyfit(js[2:],np.log2(np.array(out[2:])+1e-300),1)[0]; print('  log2-slope',round(sl,2))
np.save('e5b.npy',B,allow_pickle=True)

# (c) complexity tail of H in wavelet basis, translation-averaged multiplier sup-norm
N=1024; L=10
k=np.fft.fftfreq(N)*N; mH=-1j*np.sign(k)
def Hop(v): return np.real(np.fft.ifft(mH*np.fft.fft(v)))
for w in ['haar','db4']:
    W=np.array([np.concatenate(pywt.wavedec(e,w,mode='periodization',level=L)) for e in np.eye(N)]).T  # coeff x sample
    HM=np.array([Hop(e) for e in np.eye(N)]).T   # matrix of H
    M=W@HM@W.T
    # index -> (level, k); order: cA_L (1), cD_L(1), cD_{L-1}(2),...,cD_1(512)
    lev=[L]; kk=[0]
    for l in range(L,0,-1):
        n=N>>l; lev+= [l]*n; kk+=list(range(n))
    lev=np.array(lev); kk=np.array(kk); isA=np.zeros(N,bool); isA[0]=True
    def anc(l,kq,Lq): return (kq<<l)>>Lq
    kap=np.zeros((N,N),int)
    for a in range(N):
        la,ka=lev[a],kk[a]
        Ls=np.maximum(la,lev)
        Lstar=Ls.copy()
        for Lq in range(1,L+1):
            same=(anc(la,ka,np.maximum(Lq,Ls))==anc(lev,kk,np.maximum(Lq,Ls)))
            Lstar=np.where((Lstar==Ls)&~same&(Lq>Ls),Lq+0,Lstar)
        # compute properly: smallest Lq>=Ls with same ancestor
        Lst=np.full(N,L)
        for Lq in range(L,-1,-1):
            ok=(Lq>=Ls)&(anc(la,ka,Lq)==anc(lev,kk,Lq)) if True else None
            Lst=np.where(ok,Lq,Lst)
        kap[a]=np.maximum(Lst-la,Lst-lev)
    out=[];outfix=[]
    F=np.fft.fft(np.eye(N))/np.sqrt(N)
    for kc in range(0,9):
        T=np.where(kap>kc,M,0.0); T[isA,:]=0; T[:,isA]=0
        X=W.T@T@W
        d=np.real(np.einsum('ij,jk,ik->i',F,X,F.conj()))+1j*np.imag(np.einsum('ij,jk,ik->i',F,X,F.conj()))
        out.append(np.max(np.abs(d))); outfix.append(np.linalg.norm(X,2))
    print(w,'avg tail sup|m|:',["%.2e"%v for v in out]); print(w,'fixed-lattice tail norm:',["%.2f"%v for v in outfix])
    np.save(f'e5c_{w}.npy',np.array([out,outfix]))
