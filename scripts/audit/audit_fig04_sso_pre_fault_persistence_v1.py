#!/usr/bin/env python3
"""Read-only audit: does configured-background residual persist to fault onset?"""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np,pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'data/frozen_inputs'
OUT=REPO/'outputs/fig04_pre_fault_persistence_audit_v1'
FS=20000.;RMS_BLOCK=.02;COMP_BLOCK=.04;START=.08;END=.30
CONF={'ieee14':{'ref':7,'fbus':2,'ids':[19,20,21]},'ieee30':{'ref':30,'fbus':1,'ids':[559,560,561]}}
CONDS=[('NoSSO','No-SSO','#333333'),('SSO25Hz_M01','25 Hz, 1%','#56B4E9'),('SSO25Hz_M03','25 Hz, 3%','#E69F00')]
FONT='Times New Roman'
try:font_manager.findfont(font_manager.FontProperties(family=[FONT]),fallback_to_default=False)
except ValueError:FONT='Liberation Serif'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def component(t,x,f):
 """Exact least-squares amplitude in one 25-Hz period; source difference removes 50 Hz first."""
 a=np.column_stack([np.ones(len(t)),np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)])
 c=np.linalg.lstsq(a,x,rcond=None)[0];return float(np.hypot(c[1],c[2]))
def main():
 if OUT.exists():
  if '--force' in sys.argv:shutil.rmtree(OUT)
  else:raise SystemExit(f'Refusing to overwrite {OUT}; pass --force to replace this exact result folder')
 for x in ['png','pdf','plot_data','diagnostics']:(OUT/x).mkdir(parents=True)
 mpl.rcParams.update({'font.family':FONT,'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 m=pd.read_csv(ROOT/'manifests/fault_generalization_manifest.csv');cases={};provenance=[]
 for net,cfg in CONF.items():
  q=m[(m.NetworkID==net)&(m.FaultBus==cfg['fbus'])&(m.FaultType=='SLG')&(m.FaultResistanceOhm==10)&(m.FaultInceptionAngleDeg==0)]
  for bg,_,_ in CONDS:
   r=q[q.BackgroundName==bg]
   if len(r)!=1:raise ValueError((net,bg,len(r)))
   r=r.iloc[0];p=ROOT/'waveforms_selected'/Path(r.OutputFile).name;d=pd.read_csv(p)
   if int(r.CaseID) not in cfg['ids']:raise ValueError('unexpected case')
   cases[(net,bg)]=(r,d);provenance.append({'Network':net,'ReferenceBus':cfg['ref'],'CaseID':int(r.CaseID),'Background':bg,'RawCSV':str(p),'SHA256':sha(p)})
 rec=[];fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
 for row,net in enumerate(['ieee14','ieee30']):
  cfg=CONF[net];bus=cfg['ref'];_,base=cases[(net,'NoSSO')];t=base.Time.to_numpy(float);x0=base[f'Va_{bus}'].to_numpy(float);allpre=(t>=START)&(t<END);base50=component(t[allpre],x0[allpre],50.)
  for bg,label,color in CONDS[1:]:
   r,d=cases[(net,bg)];res=d[f'Va_{bus}'].to_numpy(float)-x0
   # Residual RMS uses contiguous 20-ms (one fundamental-cycle) blocks through fault onset.
   for left in np.arange(START,END,RMS_BLOCK):
    right=min(left+RMS_BLOCK,END);ix=(t>=left)&(t<right);xx=res[ix]
    rec.append({'Network':net,'Panel':'residual_rms','ReferenceBus':bus,'CaseID':int(r.CaseID),'Background':bg,'StartS':left,'EndS':right,'CenterS':(left+right)/2,'ResidualRMSPctOfNoSSO50Hz':100*np.sqrt(np.mean(xx**2))/base50,'Component25HzPctOfNoSSO50Hz':np.nan,'Component75HzPctOfNoSSO50Hz':np.nan})
   # Each component fit spans a full 40-ms 25-Hz period; final fit deliberately ends at onset.
   for left in [.08,.12,.16,.20,.24,.26]:
    right=left+COMP_BLOCK;ix=(t>=left)&(t<right);tt=t[ix];xx=res[ix]
    rec.append({'Network':net,'Panel':'component_fit','ReferenceBus':bus,'CaseID':int(r.CaseID),'Background':bg,'StartS':left,'EndS':right,'CenterS':(left+right)/2,'ResidualRMSPctOfNoSSO50Hz':np.nan,'Component25HzPctOfNoSSO50Hz':100*component(tt,xx,25.)/base50,'Component75HzPctOfNoSSO50Hz':100*component(tt,xx,75.)/base50})
  z=pd.DataFrame([x for x in rec if x['Network']==net])
  for bg,label,color in CONDS[1:]:
   qr=z[(z.Background==bg)&(z.Panel=='residual_rms')];qc=z[(z.Background==bg)&(z.Panel=='component_fit')];axs[row,0].plot(qr.CenterS,qr.ResidualRMSPctOfNoSSO50Hz,marker='o',color=color,lw=2,label=label);axs[row,1].plot(qc.CenterS,qc.Component25HzPctOfNoSSO50Hz,marker='o',color=color,lw=1.8,label=f'{label}: 25 Hz');axs[row,1].plot(qc.CenterS,qc.Component75HzPctOfNoSSO50Hz,marker='s',color=color,lw=1.3,ls='--',label=f'{label}: 75 Hz')
  for ax in axs[row]:
   ax.axvline(.30,color='k',ls='--',lw=.9,label='Fault onset');ax.axvspan(.26,.30,color='#EAF2F8',alpha=.5);ax.grid(alpha=.25);ax.set_xlim(START,.305)
  axs[row,0].set(title=f"{net.upper()} documented reference Bus {bus}: paired residual",ylabel='Residual RMS\n(% of No-SSO 50 Hz)')
  axs[row,1].set(title=f"{net.upper()} documented reference Bus {bus}: exact components",ylabel='Component amplitude\n(% of No-SSO 50 Hz)')
 axs[1,0].set_xlabel('Pre-fault time (s)');axs[1,1].set_xlabel('Pre-fault time (s)');axs[0,0].legend(frameon=False);axs[0,1].legend(frameon=False,ncol=2,fontsize=8)
 fig.savefig(OUT/'png/fig04_pre_fault_persistence.png',dpi=600,bbox_inches='tight');fig.savefig(OUT/'pdf/fig04_pre_fault_persistence.pdf',bbox_inches='tight');plt.close(fig)
 df=pd.DataFrame(rec);df.to_csv(OUT/'plot_data/pre_fault_persistence.csv',index=False);pd.DataFrame(provenance).to_csv(OUT/'diagnostics/selected_cases_and_hashes.csv',index=False)
 rms=df[df.Panel.eq('residual_rms')].sort_values('EndS').groupby(['Network','Background']).tail(1).set_index(['Network','Background'])['ResidualRMSPctOfNoSSO50Hz']
 comp=df[df.Panel.eq('component_fit')].sort_values('EndS').groupby(['Network','Background']).tail(1).set_index(['Network','Background'])[['Component25HzPctOfNoSSO50Hz','Component75HzPctOfNoSSO50Hz']]
 peak=df[df.Panel.eq('residual_rms')].groupby(['Network','Background'])['ResidualRMSPctOfNoSSO50Hz'].max()
 summary=pd.concat([rms.rename('Last20msResidualRMSPct'),comp.rename(columns={'Component25HzPctOfNoSSO50Hz':'Trailing40ms25HzPct','Component75HzPctOfNoSSO50Hz':'Trailing40ms75HzPct'}),peak.rename('Max20msResidualRMSPct')],axis=1).reset_index();summary.to_csv(OUT/'diagnostics/persistence_summary.csv',index=False)
 meta={'scope':'read-only audit of frozen current 1,080-case raw corpus; no simulation or raw regeneration','window_s':[START,END],'rms_block_s':RMS_BLOCK,'component_block_s':COMP_BLOCK,'fault_onset_s':.3,'bus_verdict':{'ieee14':'Bus 7, current corpus execution lineage','ieee30':'Bus 30, documented execution provenance only; physical topology unverified'},'provenance':provenance,'font':FONT};(OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2));(OUT/'README.md').write_text('# SSO pre-fault persistence audit\n\nPaired SSO-minus-NoSSO voltage residual at the documented execution reference bus. Residual RMS uses one 50-Hz cycle (20 ms) including the final 0.28–0.30 s block. Frequency components use complete 40-ms windows; the final fit is 0.26–0.30 s ending at configured fault onset. This is an observation of archived waveform behavior, not a physical SSO/PCC-topology validation.\n')
 print(OUT);print(summary.to_string(index=False))
if __name__=='__main__':main()
