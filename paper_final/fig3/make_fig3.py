# Figure 3 (fig:deform) from e3_aligned.json: Haar, aligned sym4 (m_d=0, m=3), aligned db4 (m_d=1, m=4).
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B,O,A,G='#2a78d6','#eb6834','#1baf7a','#52514e'   # paper palette (validated: CVD dE 9.2, green labelled directly)
plt.rcParams.update({'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9994',
 'axes.grid':True,'grid.color':'#e6e5e0','grid.linewidth':0.6,'lines.linewidth':1.6,'legend.frameon':False,'figure.dpi':200})
o=json.load(open('e3_aligned.json'))
eps=np.array(o['eps']); P=np.array(o['P']); tinf=0.02*P/(2*np.pi)
gens=[('haar','Haar',B,'o'),('sym4','sym4 ($m=3$)',A,'s'),('db4','db4 ($m=4$)',O,'^')]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.75))
for k,lab,c,m in gens:
    a,s=o[k+'_eps']; axs[0].loglog(eps,a,'-'+m,color=c,ms=4.5,label=f'{lab}, lattice average')
    axs[0].loglog(eps,s,'--'+m,color=c,ms=4.5,mfc='white',label=f'{lab}, single lattice')
    a,s=o[k+'_P']; axs[1].loglog(tinf,a,'-'+m,color=c,ms=4.5); axs[1].loglog(tinf,s,'--'+m,color=c,ms=4.5,mfc='white')
# slope-1 guide
axs[0].loglog(eps,0.5*np.array(o['haar_eps'][0][0])*eps/eps[0],':',color=G,lw=1)
axs[0].text(eps[2],0.5*o['haar_eps'][0][0]*eps[2]/eps[0]*0.62,'slope 1',color=G,fontsize=7)
axs[0].set_xticks([0.0025,0.01,0.04]); axs[0].set_xticklabels(['0.0025','0.01','0.04']); axs[0].minorticks_off()
axs[0].set_xlabel(r"$\|\tau^\prime\|_\infty$  (period $P=N_{\rm s}$)"); axs[0].set_ylabel(r"covariance error / $\|f\|$")
axs[1].set_xticks([1,3,10,30]); axs[1].set_xticklabels(['1','3','10','30']); axs[1].minorticks_off()
axs[1].set_xlabel(r'$\|\tau\|_\infty$ (samples), $\|\tau^\prime\|_\infty=0.02$ fixed')
for ax in axs: ax.set_ylim(1.5e-4,1.5)
axs[0].set_title(r'(a) dependence on $\|\tau^\prime\|_\infty$',fontsize=8.5); axs[1].set_title(r'(b) dependence on $\|\tau\|_\infty$',fontsize=8.5)
h,l=axs[0].get_legend_handles_labels()
order=[0,1,2,3,4,5]
fig.legend([h[i] for i in order],[l[i] for i in order],loc='lower center',ncol=3,fontsize=7,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.13,1,1)); fig.savefig('fig3_en.pdf'); fig.savefig('fig3_en.png',dpi=150)
