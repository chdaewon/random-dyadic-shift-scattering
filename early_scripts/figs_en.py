import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B,O,A,G='#2a78d6','#eb6834','#1baf7a','#52514e'
plt.rcParams.update({'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9994',
 'axes.grid':True,'grid.color':'#e6e5e0','grid.linewidth':0.6,'lines.linewidth':2,'legend.frameon':False,'figure.dpi':200})
ex=np.pi/(8*np.log(2))
# Fig1 ripple
s,re,im=np.load('MT_haar.npy'); MT=im
xi,mre,mim=np.load('e1.npy')
fig,ax=plt.subplots(figsize=(6,3.1))
ss=np.concatenate([s,s+1,s+2]); ax.plot(ss,np.tile(MT,3),color=B,label='translation average only (continuous formula)')
sel=(xi>8)&(xi<64); ax.plot(np.log2(xi[sel])-3,mim[sel],'o',ms=4,color=B,mfc='white',label='discrete implementation (N=4096, exhaustive)')
for Q,c,ls in [(2,O,'--'),(4,A,':')]:
    MQ=np.array([np.mean([MT[(i+q*40//Q)%40] for q in range(Q)]) for i in range(40)]); ax.plot(ss,np.tile(MQ,3),ls,color=c,label=f'+ {Q}-point dilation quadrature')
ax.axhline(ex,color=G,lw=1); ax.text(3.02,ex,r'$\pi/(8\ln 2)$',va='center',color=G,fontsize=8)
ax.set_xlabel(r'$\log_2\xi$'); ax.set_ylabel(r'Im $m(\xi)$'); ax.set_xlim(0,3); ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=2)
fig.tight_layout(); fig.savefig('fig1_en.pdf')
# Fig2 per-sample translation
a,h,d2,d4=np.load('e2.npy')
fig,ax=plt.subplots(figsize=(3.2,2.6))
for y,c,m,l in [(h,B,'o','Haar'),(d2,A,'s','db2'),(d4,O,'^','db4')]: ax.loglog(a,y,'-'+m,color=c,ms=5,label=l)
aa=a[:3]; ax.loglog(aa,1.35*h[0]*(aa/aa[0])**0.5,':',color=B,lw=1); ax.loglog(aa,0.7*d4[0]*(aa/aa[0]),':',color=O,lw=1)
ax.text(aa[0],1.35*h[0]*1.12,r'$\propto a^{1/2}$',fontsize=7,color=B); ax.text(aa[1],0.7*d4[1]*0.75,r'$\propto a$',fontsize=7,color=O)
ax.set_xticks(a); ax.set_xticklabels(['1/32','1/16','1/8','1/4','1/2']); ax.set_yticks([0.4,0.7,1,1.5]); ax.set_yticklabels(['0.4','0.7','1','1.5']); ax.minorticks_off()
ax.set_xlabel(r'$2^J|a|$ (shift relative to finest cell)'); ax.set_ylabel(r'$\|S_\omega-S_{\omega+a}\|$'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig2_en.pdf')
# Fig3 deformation
o=np.load('e3.npy',allow_pickle=True).item()
eps=[0.0025,0.005,0.01,0.02,0.04,0.08]; P=np.array([256,512,1024,2048,4096,8192]); tinf=0.02*P/(2*np.pi)
fig,axs=plt.subplots(1,2,figsize=(6.4,2.6))
for w,c,m in [('haar',B,'o'),('db4',O,'^')]:
    A_,S_=o[w+'_eps']; axs[0].loglog(eps,A_,'-'+m,color=c,ms=4,label=f'{w}: lattice average'); axs[0].loglog(eps,S_,'--'+m,color=c,ms=4,mfc='white',label=f'{w}: single lattice')
    A_,S_=o[w+'_P']; axs[1].semilogx(tinf,A_,'-'+m,color=c,ms=4); axs[1].semilogx(tinf,S_,'--'+m,color=c,ms=4,mfc='white')
axs[0].set_xticks([0.0025,0.01,0.04]); axs[0].set_xticklabels(['0.0025','0.01','0.04']); axs[0].minorticks_off(); axs[0].set_xlabel(r'$\|\tau^\prime\|_\infty$'); axs[0].set_ylabel('covariance error / $\\|f\\|$'); axs[0].legend(fontsize=6.5)
axs[1].set_xlabel(r'$\|\tau\|_\infty$ (samples), $\|\tau^\prime\|_\infty=0.02$ fixed'); axs[1].set_yscale('log')
axs[0].set_title('(a) dependence on $\\|\\tau^\\prime\\|_\\infty$',fontsize=8.5); axs[1].set_title('(b) dependence on $\\|\\tau\\|_\\infty$',fontsize=8.5)
fig.tight_layout(); fig.savefig('fig3_en.pdf')

# Fig4 integrand
g,h,d0,d4=np.load('e9_int.npy')
fig,ax=plt.subplots(figsize=(6,2.6))
ax.plot(g,h,color=B,label='Haar ($\\kappa=0.567i$)')
ax.plot(g,d0,'--',color=O,label='db4, naive children $m=0$ ($\\kappa\\approx0.011+0.023i$)')
ax.plot(g,-d4,':',color=A,lw=2.4,label='db4, aligned $m=4$, sign flipped ($|\\kappa|\\approx0.457$)')
ax.axhline(0,color=G,lw=0.8)
ax.set_xlabel(r'$\log_2 t$'); ax.set_ylabel(r'Im $m_0(t)$'); ax.set_xlim(-2.5,2.5)
ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=2)
fig.tight_layout(); fig.savefig('fig4_en.pdf')
