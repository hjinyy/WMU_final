#!/usr/bin/env python3
"""Waveform-only onset plus paired C1-C4 features for the 4,752-case corpus.

Detection never receives manifest labels/timestamps. Manifest onset is used only
for the explicitly named reference-window arm and post-hoc scoring.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,math,sys,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

REPO=Path('/home/hy/WMU_project')
import importlib.util
_WAVEFORM_PATH=REPO/'src/wmu_project/hybrid_fault_v3/waveform.py'
_WAVEFORM_SPEC=importlib.util.spec_from_file_location('wmu_step2_waveform',_WAVEFORM_PATH)
if _WAVEFORM_SPEC is None or _WAVEFORM_SPEC.loader is None: raise ImportError(f'cannot load {_WAVEFORM_PATH}')
_WAVEFORM_MODULE=importlib.util.module_from_spec(_WAVEFORM_SPEC)
sys.modules[_WAVEFORM_SPEC.name]=_WAVEFORM_MODULE
_WAVEFORM_SPEC.loader.exec_module(_WAVEFORM_MODULE)
detect_event_multi=_WAVEFORM_MODULE.detect_event_multi

AUDIT=REPO/'results/fault_generalization_step1_audit_v1'
MANIFEST=AUDIT/'integrated_manifest_4752_final.csv'
BUS_COUNTS={'ieee14':14,'ieee30':30}
DETECT_WMUS=(6,1,2)
FS0=50.0; EPS=1e-12
PRE_DURATION,EVENT_DURATION,POST_DURATION=.10,3.0/FS0,.10
A=np.exp(2j*np.pi/3)

C1_FEATURES=['pmu_V1','pmu_V2_over_V1','pmu_V0_over_V1','pmu_I1','pmu_I2_over_I1','pmu_I0_over_I1']
C2_FEATURES=[]
for q in ('v','i'):
    for phase in 'ABC': C2_FEATURES += [f'{q}_pre_rms_{phase}',f'{q}_event_rms_{phase}',f'{q}_post_rms_{phase}']
C2_FEATURES += ['voltage_sag_ratio','voltage_phase_rms_mean','voltage_phase_rms_std','event_min_voltage','current_jump_ratio','current_phase_rms_mean','current_phase_rms_std','event_max_current','pre_to_event_voltage_change','pre_to_event_current_change','pre_to_post_voltage_change','pre_to_post_current_change']
C2_FEATURES += ['V1','V2_over_V1','V0_over_V1','I1','I2_over_I1','I0_over_I1','sso_frequency_energy','lowfreq_5_45_energy','fundamental_magnitude','dominant_lowfreq_component']
C3_FEATURES=['I_rel_sum','I_rel_max','V_rel_sum','V_rel_max']
C4_FEATURES=[]
for q in ('v','i'):
    for phase in 'abc': C4_FEATURES += [f'{q}_p2p_over_pre_rms_{phase}',f'{q}_derivative_rms_over_pre_rms_{phase}']
ALL_FEATURES=list(dict.fromkeys(C1_FEATURES+C2_FEATURES+C3_FEATURES+C4_FEATURES))
META_FEATURE_FIELDS=['WindowMode','UnifiedCaseID','NetworkID','CaseID','SourceDataset','FaultType','FaultBusLabel','FaultResistanceOhm','FaultInceptionAngleDeg','BackgroundName','MeasuredBusLabel','DetectionStatus','DetectedOnsetS','DecisionAvailableS','TruthOnsetS_posthoc','OnsetSignedErrorMs_posthoc','FeatureWindowStartS','FeatureWindowEndS','RawCSV_SHA256']


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def write_csv(path:Path,rows:list[dict],fields:list[str])->None:
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
def netkey(v): return str(v).strip().lower()
def bus_count(v): return BUS_COUNTS[netkey(v)]
def seq(ph):
    va,vb,vc=ph
    return ((va+A*vb+A*A*vc)/3,(va+A*A*vb+A*vc)/3,(va+vb+vc)/3)
def phasor(t,x,basis=None,f=FS0):
    if basis is None:
        w=2*np.pi*f;basis=(2.0/len(x)*np.cos(w*t),2.0/len(x)*np.sin(w*t))
    c=np.dot(x,basis[0]);s=np.dot(x,basis[1])
    return (c-1j*s)/np.sqrt(2.0)
def legacy_baseline_phasor(x,dt):
    x=np.asarray(x,dtype=float);t=np.arange(len(x),dtype=float)*float(dt)
    return complex(np.sum(x*np.exp(-1j*2*np.pi*FS0*t))/len(x)) if len(x) else 0j
def safe_ratio(a,b):
    a=float(a);b=float(b)
    return 0.0 if not math.isfinite(a) or not math.isfinite(b) or abs(b)<=EPS else float(a/b)
def rms(a): return float(np.sqrt(np.mean(np.asarray(a,dtype=float)**2)))
def col_index(bus,phase,kind):
    pi={'a':0,'b':1,'c':2}[phase]
    return 1+(bus-1)*6+pi+(3 if kind=='i' else 0)
def load_waveform(path:Path,nbus:int):
    payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
    first=payload.splitlines()[0].decode('utf-8','replace').strip()
    header=first.split(',',1)[0].strip().lower().startswith('time')
    arr=np.loadtxt(io.BytesIO(payload),delimiter=',',skiprows=1 if header else 0,dtype=np.float64)
    if arr.ndim!=2 or arr.shape!=(10001,1+6*nbus): raise ValueError(f'raw_shape_{arr.shape}_expected_10001x{1+6*nbus}')
    return arr,digest

def spectral_features(arr:np.ndarray,t:np.ndarray,spectmask:np.ndarray,nbus:int,sso_hz:float)->list[dict[str,float]]:
    cols=[col_index(b,p,'v') for b in range(1,nbus+1) for p in 'abc']
    v=arr[np.ix_(spectmask,cols)].reshape(int(spectmask.sum()),nbus,3)
    vm=np.sqrt(np.sum(v*v,axis=2));x=vm-np.mean(vm,axis=0,keepdims=True)
    power=np.abs(np.fft.rfft(x,axis=0))**2;freq=np.fft.rfftfreq(len(x),d=float(np.median(np.diff(t))))
    band=(freq>=5)&(freq<=45);total=(freq>0)&(freq<=100);band_idx=np.flatnonzero(band)
    low=np.sum(power[band,:],axis=0)/np.maximum(np.sum(power[total,:],axis=0),EPS)
    around=(freq>=max(0.0,float(sso_hz)-2.0))&(freq<=float(sso_hz)+2.0) if float(sso_hz)>0 else np.zeros_like(freq,dtype=bool)
    sso=np.sum(power[around,:],axis=0)/np.maximum(np.sum(power[total,:],axis=0),EPS)
    dom=freq[band_idx[np.argmax(power[band,:],axis=0)]] if len(band_idx) else np.zeros(nbus)
    fund=np.sqrt(power[int(np.argmin(abs(freq-50.0))),:])
    return [{'sso_frequency_energy':float(sso[j]),'lowfreq_5_45_energy':float(low[j]),'dominant_lowfreq_component':float(dom[j]),'fundamental_magnitude':float(fund[j])} for j in range(nbus)]

def window_masks(t:np.ndarray,anchor:float):
    dt=float(np.median(np.diff(t)));npre=int(round(PRE_DURATION/dt));nevent=int(round(EVENT_DURATION/dt));npost=int(round(POST_DURATION/dt))
    start_idx=int(round((float(anchor)-float(t[0]))/dt))
    if start_idx<0 or start_idx>=len(t) or abs(float(t[start_idx])-float(anchor))>dt*.51:return None
    lo=start_idx-npre;event_end=start_idx+nevent;post_start=event_end+1;hi=post_start+npost
    if lo<0 or hi>len(t):return None
    pre=np.zeros(len(t),dtype=bool);ev=np.zeros(len(t),dtype=bool);post=np.zeros(len(t),dtype=bool)
    pre[lo:start_idx]=True;ev[start_idx:event_end+1]=True;post[post_start:hi]=True
    return pre,ev,post,float(t[lo]),float(t[hi-1])

def raw_features(arr:np.ndarray,t:np.ndarray,premask:np.ndarray,evmask:np.ndarray,postmask:np.ndarray,bus:int,basis,spec)->dict[str,float]:
    spectmask=premask|evmask|postmask
    tw=t[evmask];dt=float(np.median(np.diff(t)));vpre=[];vev=[];vpost=[];ipre=[];iev=[];ipost=[];vph=[];iph=[];vph_base=[];iph_base=[];channels={};vmins=[];imaxs=[]
    for phase in 'abc':
        v=arr[:,col_index(bus,phase,'v')];i=arr[:,col_index(bus,phase,'i')]
        vp=v[premask];ve=v[evmask];vpo=v[postmask];ip=i[premask];ie=i[evmask];ipo=i[postmask]
        vrp=rms(vp);vre=rms(ve);vrpo=rms(vpo);irp=rms(ip);ire=rms(ie);irpo=rms(ipo)
        vpre.append(vrp);vev.append(vre);vpost.append(vrpo);ipre.append(irp);iev.append(ire);ipost.append(irpo)
        vph.append(phasor(tw,ve,basis));iph.append(phasor(tw,ie,basis))
        vph_base.append(legacy_baseline_phasor(ve,dt));iph_base.append(legacy_baseline_phasor(ie,dt))
        channels[(phase,'v')]=(vp,ve,vrp);channels[(phase,'i')]=(ip,ie,irp)
        vmins.append(float(np.min(np.abs(ve))));imaxs.append(float(np.max(np.abs(ie))))
    pmuv=seq(vph);pmui=seq(iph)
    pmuv1,pmuv2,pmuv0=abs(pmuv[0]),abs(pmuv[1]),abs(pmuv[2]);pmui1,pmui2,pmui0=abs(pmui[0]),abs(pmui[1]),abs(pmui[2])
    basev=seq(vph_base);basei=seq(iph_base)
    v1,v2,v0=abs(basev[0]),abs(basev[1]),abs(basev[2]);i1,i2,i0=abs(basei[0]),abs(basei[1]),abs(basei[2])
    out={'pmu_V1':pmuv1,'pmu_V2_over_V1':safe_ratio(pmuv2,pmuv1),'pmu_V0_over_V1':safe_ratio(pmuv0,pmuv1),'pmu_I1':pmui1,'pmu_I2_over_I1':safe_ratio(pmui2,pmui1),'pmu_I0_over_I1':safe_ratio(pmui0,pmui1),
      'V1':v1,'V2_over_V1':safe_ratio(v2,v1),'V0_over_V1':safe_ratio(v0,v1),'I1':i1,'I2_over_I1':safe_ratio(i2,i1),'I0_over_I1':safe_ratio(i0,i1),
      'sso_frequency_energy':spec['sso_frequency_energy'],'lowfreq_5_45_energy':spec['lowfreq_5_45_energy'],'dominant_lowfreq_component':spec['dominant_lowfreq_component'],'fundamental_magnitude':spec['fundamental_magnitude'],
      'voltage_sag_ratio':safe_ratio(float(np.mean(vpre))-float(np.min(vev)),float(np.mean(vpre))),'voltage_phase_rms_mean':float(np.mean(vev)),'voltage_phase_rms_std':float(np.std(vev)),'event_min_voltage':float(np.min(vmins)),
      'current_jump_ratio':safe_ratio(float(np.mean(iev))-float(np.mean(ipre)),float(np.mean(ipre))),'current_phase_rms_mean':float(np.mean(iev)),'current_phase_rms_std':float(np.std(iev)),'event_max_current':float(max(np.max(np.abs(arr[evmask,col_index(bus,p,'i')])) for p in 'abc')),
      'pre_to_event_voltage_change':float(np.mean(vev)-np.mean(vpre)),'pre_to_event_current_change':float(np.mean(iev)-np.mean(ipre)),'pre_to_post_voltage_change':float(np.mean(vpost)-np.mean(vpre)),'pre_to_post_current_change':float(np.mean(ipost)-np.mean(ipre))}
    for j,phase in enumerate('ABC'):
        out[f'v_pre_rms_{phase}']=vpre[j];out[f'v_event_rms_{phase}']=vev[j];out[f'v_post_rms_{phase}']=vpost[j];out[f'i_pre_rms_{phase}']=ipre[j];out[f'i_event_rms_{phase}']=iev[j];out[f'i_post_rms_{phase}']=ipost[j]
    for phase in 'abc':
        for q in ('v','i'):
            pre,ev,base=channels[(phase,q)]
            out[f'{q}_p2p_over_pre_rms_{phase}']=safe_ratio(float(np.ptp(ev)),base)
            out[f'{q}_derivative_rms_over_pre_rms_{phase}']=safe_ratio(rms(np.diff(ev)/dt),base)
    if not all(math.isfinite(float(v)) for v in out.values()): raise ValueError('nonfinite_extracted_feature')
    return out

def enrich_normvi(bus_features:list[dict[str,float]])->None:
    dv=np.array([abs(r['voltage_phase_rms_mean']-float(np.mean([r[f'v_pre_rms_{p}'] for p in 'ABC']))) for r in bus_features])
    di=np.array([abs(r['current_phase_rms_mean']-float(np.mean([r[f'i_pre_rms_{p}'] for p in 'ABC']))) for r in bus_features])
    for j,r in enumerate(bus_features):
        r['V_rel_sum']=float(dv[j]/(float(np.sum(dv))+EPS));r['V_rel_max']=float(dv[j]/(float(np.max(dv))+EPS))
        r['I_rel_sum']=float(di[j]/(float(np.sum(di))+EPS));r['I_rel_max']=float(di[j]/(float(np.max(di))+EPS))

def detect_only(arr:np.ndarray,t:np.ndarray,nbus:int):
    vs=[];is_=[]
    for bus in DETECT_WMUS:
        vs.append(np.column_stack([arr[:,col_index(bus,p,'v')] for p in 'abc']))
        is_.append(np.column_stack([arr[:,col_index(bus,p,'i')] for p in 'abc']))
    # No manifest fields are passed; this detector receives raw synchronized WMU V/I only.
    return detect_event_multi(t,vs,is_)

def rows_for_window(arr,t,row,window_mode,anchor,detection_status,onset,available,error_ms):
    masks=window_masks(t,float(anchor))
    if masks is None:return None,'windows_out_of_bounds_or_anchor_off_grid'
    premask,evmask,postmask,start,end=masks;spectmask=premask|evmask|postmask;nbus=bus_count(row['NetworkID']);dt=float(np.median(np.diff(t)))
    expected=(int(round(PRE_DURATION/dt)),int(round(EVENT_DURATION/dt))+1,int(round(POST_DURATION/dt)))
    if (premask.sum(),evmask.sum(),postmask.sum())!=expected:return None,'wrong_window_sample_count'
    basis=(2.0/evmask.sum()*np.cos(2*np.pi*FS0*t[evmask]),2.0/evmask.sum()*np.sin(2*np.pi*FS0*t[evmask]))
    specs=spectral_features(arr,t,spectmask,nbus,float(row.get('SSOFrequencyHz',0) or 0))
    features=[raw_features(arr,t,premask,evmask,postmask,bus,basis,specs[bus-1]) for bus in range(1,nbus+1)]
    enrich_normvi(features);records=[]
    for bus,feat in enumerate(features,1):
        m={'WindowMode':window_mode,'UnifiedCaseID':int(row['UnifiedCaseID']),'NetworkID':row['NetworkID'],'CaseID':int(row['SourceCaseID']),'SourceDataset':row['SourceDataset'],'FaultType':row['FaultType'],'FaultBusLabel':int(row['FaultBus']),'FaultResistanceOhm':float(row['FaultResistanceOhm']),'FaultInceptionAngleDeg':int(row['FaultInceptionAngleDeg']),'BackgroundName':row['BackgroundName'],'MeasuredBusLabel':bus,'DetectionStatus':detection_status,'DetectedOnsetS':onset,'DecisionAvailableS':available,'TruthOnsetS_posthoc':float(row['EventStartTime']),'OnsetSignedErrorMs_posthoc':error_ms,'FeatureWindowStartS':start,'FeatureWindowEndS':end,'RawCSV_SHA256':row['ActualRawFileSHA256']}
        records.append({**m,**feat})
    return records,'PASS'

def write_gzip_features(path:Path,rows,fields):
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
        for record in rows:w.writerow(record)

def grouped_detection(good,key):
    groups=defaultdict(list)
    for r in good:groups[r[key]].append(r)
    out=[]
    for value,items in sorted(groups.items(),key=lambda kv:str(kv[0])):
        errs=[abs(float(x['OnsetSignedErrorMs_posthoc'])) for x in items if x.get('OnsetSignedErrorMs_posthoc') not in ('',None)]
        detected=sum(x['DetectorStatus']=='DETECTED' for x in items)
        out.append({'GroupField':key,'GroupValue':value,'Cases':len(items),'Detected':detected,'Coverage':detected/len(items) if items else None,'Within5ms':sum(e<=5 for e in errs),'Within10ms':sum(e<=10 for e in errs),'Within20ms':sum(e<=20 for e in errs),'MedianAbsErrorMs_detected':float(np.median(errs)) if errs else None,'P95AbsErrorMs_detected':float(np.quantile(errs,.95)) if errs else None})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--case-ids',default='');a=ap.parse_args()
    target=Path(a.output).resolve()
    if target.exists():raise FileExistsError(f'refusing to overwrite {target}')
    target.mkdir(parents=True)
    with MANIFEST.open(encoding='utf-8',newline='') as f: all_rows=list(csv.DictReader(f))
    if len(all_rows)!=4752:raise SystemExit(f'expected 4,752 manifest rows; found {len(all_rows)}')
    if a.case_ids:
        wanted={int(x) for x in a.case_ids.split(',') if x.strip()};rows=[r for r in all_rows if int(r['UnifiedCaseID']) in wanted]
        if {int(r['UnifiedCaseID']) for r in rows}!=wanted:raise SystemExit('requested CaseID not found')
    else:rows=all_rows[:a.limit] if a.limit else all_rows
    pred=[];errors=[];feature_status=[];count_by_mode=Counter();clock=time.monotonic();fields=META_FEATURE_FIELDS+ALL_FEATURES
    detected_path=target/'features_detected.csv.gz';oracle_path=target/'features_manifest_reference.csv.gz'
    with gzip.open(detected_path,'wt',encoding='utf-8',newline='') as fd,gzip.open(oracle_path,'wt',encoding='utf-8',newline='') as fo:
        wd=csv.DictWriter(fd,fieldnames=fields,extrasaction='ignore',lineterminator='\n');wo=csv.DictWriter(fo,fieldnames=fields,extrasaction='ignore',lineterminator='\n');wd.writeheader();wo.writeheader()
        for n,r in enumerate(rows,1):
            cid=int(r['UnifiedCaseID']);base={'UnifiedCaseID':cid,'SourceDataset':r['SourceDataset'],'NetworkID':r['NetworkID'],'CaseID':int(r['SourceCaseID']),'FaultType':r['FaultType'],'FaultBusLabel':int(r['FaultBus']),'FaultResistanceOhm':float(r['FaultResistanceOhm']),'FaultInceptionAngleDeg':int(r['FaultInceptionAngleDeg']),'BackgroundName':r['BackgroundName'],'TruthOnsetS_posthoc':float(r['EventStartTime']),'RawCSV_SHA256':r['ActualRawFileSHA256']}
            try:
                nbus=bus_count(r['NetworkID']);arr,raw_digest=load_waveform(Path(r['RawCSVPath']),nbus)
                if raw_digest!=r['ActualRawFileSHA256']:raise ValueError('raw_sha256_changed_since_step1_audit')
                t=arr[:,0];dt=float(np.median(np.diff(t)))
                if not np.isfinite(arr).all():raise ValueError('raw_nonfinite')
                if not np.all(np.diff(t)>0) or abs(dt-5e-5)>1e-10 or abs(t[-1]-.5)>1e-8:raise ValueError('raw_time_grid_mismatch')
                vcols=[col_index(b,p,'v') for b in range(1,nbus+1) for p in 'abc']
                raw_max_v=float(np.max(np.abs(arr[:,vcols])))
                d=detect_only(arr,t,nbus);onset=d.onset_time_s
                # D1 rescales channels using the full record and uses a centered score;
                # it is an offline timestamp, not a causal alarm/availability time.
                available=None
                truth=float(r['EventStartTime']);err=None if onset is None else (float(onset)-truth)*1000;status='DETECTED' if d.status=='Detected' else 'NO_EVENT_DETECTED'
                pred.append({**base,'Detector':'D1_v3_MultiWMU_raw_only','DetectorStatus':status,'DetectorScore':float(d.score),'DetectorThreshold':float(d.threshold),'DetectedOnsetS':onset,'DecisionAvailableS':available,'CenteredScoreLookaheadS':.0075,'DecisionTimeStatus':'OFFLINE_FULL_RECORD_SCALE','OnsetSignedErrorMs_posthoc':err,'OnsetAbsErrorMs_posthoc':None if err is None else abs(err),'Within5ms_posthoc':None if err is None else abs(err)<=5,'Within10ms_posthoc':None if err is None else abs(err)<=10,'Within20ms_posthoc':None if err is None else abs(err)<=20,'RawMaxAbsVoltage':raw_max_v,'RawOnlyVoltageRunawayFlag_gt_1e6':raw_max_v>1e6,'RawHashMatchStep1':True,'ReadStatus':'PASS'})
                for mode,anchor,writer in [('manifest_reference',truth,wo),('detected',onset,wd)]:
                    if anchor is None:
                        feature_status.append({**base,'WindowMode':mode,'WindowStatus':'NO_DETECTION','FeatureWindowStartS':None,'FeatureWindowEndS':None});continue
                    featrows,wstatus=rows_for_window(arr,t,r,mode,float(anchor),status,onset,available,err)
                    feature_status.append({**base,'WindowMode':mode,'WindowStatus':wstatus,'FeatureWindowStartS':None if featrows is None else featrows[0]['FeatureWindowStartS'],'FeatureWindowEndS':None if featrows is None else featrows[0]['FeatureWindowEndS']})
                    if featrows is not None:writer.writerows(featrows);count_by_mode[mode]+=len(featrows)
            except Exception as e:
                errors.append({**base,'ErrorType':type(e).__name__,'Error':str(e)})
                pred.append({**base,'Detector':'D1_v3_MultiWMU_raw_only','DetectorStatus':'RAW_READ_OR_DETECT_ERROR','ReadStatus':'ERROR','Error':str(e)})
                for mode in ('manifest_reference','detected'):feature_status.append({**base,'WindowMode':mode,'WindowStatus':'RAW_READ_OR_DETECT_ERROR','Error':str(e)})
            if n%100==0:print(f'processed {n}/{len(rows)} errors={len(errors)} detected={sum(x.get("DetectorStatus")=="DETECTED" for x in pred)} elapsed={time.monotonic()-clock:.1f}s',flush=True)
    write_csv(target/'onset_predictions.csv',pred,['UnifiedCaseID','SourceDataset','NetworkID','CaseID','FaultType','FaultBusLabel','FaultResistanceOhm','FaultInceptionAngleDeg','BackgroundName','TruthOnsetS_posthoc','RawCSV_SHA256','RawHashMatchStep1','RawMaxAbsVoltage','RawOnlyVoltageRunawayFlag_gt_1e6','Detector','DetectorStatus','DetectorScore','DetectorThreshold','DetectedOnsetS','DecisionAvailableS','CenteredScoreLookaheadS','DecisionTimeStatus','OnsetSignedErrorMs_posthoc','OnsetAbsErrorMs_posthoc','Within5ms_posthoc','Within10ms_posthoc','Within20ms_posthoc','ReadStatus','Error'])
    write_csv(target/'window_feature_status.csv',feature_status,['UnifiedCaseID','SourceDataset','NetworkID','CaseID','FaultType','FaultBusLabel','FaultResistanceOhm','FaultInceptionAngleDeg','BackgroundName','TruthOnsetS_posthoc','RawCSV_SHA256','WindowMode','WindowStatus','FeatureWindowStartS','FeatureWindowEndS','Error'])
    write_csv(target/'raw_processing_errors.csv',errors,['UnifiedCaseID','SourceDataset','NetworkID','CaseID','FaultType','FaultBusLabel','RawCSV_SHA256','ErrorType','Error'])
    good=[x for x in pred if x.get('ReadStatus')=='PASS']
    groups=[]
    for k in ('NetworkID','FaultType','FaultResistanceOhm','FaultInceptionAngleDeg','BackgroundName'):groups.extend(grouped_detection(good,k))
    write_csv(target/'onset_summary_by_condition.csv',groups,['GroupField','GroupValue','Cases','Detected','Coverage','Within5ms','Within10ms','Within20ms','MedianAbsErrorMs_detected','P95AbsErrorMs_detected'])
    wcounts=Counter((x.get('WindowMode'),x.get('WindowStatus')) for x in feature_status)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'status':'COMPLETE' if len(rows)==4752 and len(good)==4752 and not errors else ('SMOKE_COMPLETE' if len(rows)<4752 and len(good)==len(rows) else 'PARTIAL_WITH_ERRORS'),'cases_requested':len(rows),'cases_read':len(good),'raw_processing_errors':len(errors),'manifest_sha256':sha256_file(MANIFEST),'detector_source_sha256':sha256_file(REPO/'src/wmu_project/hybrid_fault_v3/waveform.py'),'feature_extractor_sha256':sha256_file(Path(__file__)),'raw_only_detector':'D1_v3_MultiWMU; detector receives only raw aligned V/I channels at measured channel labels 6,1,2. Manifest timing, labels and SSO fields are not passed. It uses full-record channel-median scaling and a centered score; onset timestamps are offline detections, and no causal decision latency is claimed. The score kernel alone has 7.5-ms look-ahead.','truth_use':'EventStartTime is used only in the manifest_reference window arm and post-hoc onset error scoring; it is not passed to detect_only or raw_features.','feature_window':{'pre_event_duration_s':PRE_DURATION,'event_duration_s':EVENT_DURATION,'post_event_duration_s':POST_DURATION,'event_duration_basis':'3 cycles / 50-Hz fundamental, fixed for every case; no per-case clear-time metadata used','event_end_inclusive':True,'system_frequency_hz':FS0,'context_window_recorded':'FeatureWindowStartS/EndS span pre+event+post context'},'feature_sets':{'C1_modified_PMU_like':C1_FEATURES,'C2_WMU_Baseline':C2_FEATURES,'C3_Baseline_plus_NormVI':C2_FEATURES+C3_FEATURES,'C4_C3_plus_time_domain':C2_FEATURES+C3_FEATURES+C4_FEATURES},'feature_selector_dependencies':{'SSOFrequencyHz':'Used only to reproduce the existing baseline sso_frequency_energy spectral-band selector; not included as an ML feature and not passed to the onset detector.'},'feature_rows_by_mode':dict(count_by_mode),'window_status_counts':{f'{m}|{s}':n for (m,s),n in sorted(wcounts.items())},'detector_counts':{'detected':sum(x.get('DetectorStatus')=='DETECTED' for x in good),'not_detected':sum(x.get('DetectorStatus')=='NO_EVENT_DETECTED' for x in good),'within_5ms':sum(x.get('Within5ms_posthoc') is True for x in good),'within_10ms':sum(x.get('Within10ms_posthoc') is True for x in good),'within_20ms':sum(x.get('Within20ms_posthoc') is True for x in good),'raw_only_voltage_runaway_gt_1e6':sum(bool(x.get('RawOnlyVoltageRunawayFlag_gt_1e6')) for x in good)},'window_status_counts_by_mode':{f'{m}|{s}':n for (m,s),n in sorted(wcounts.items())},'raw_processing_errors_count':len(errors),'processing_seconds':time.monotonic()-clock}
    (target/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (target/'run_metadata.json').write_text(json.dumps({'status':summary['status'],'created_utc':summary['created_utc'],'input_manifest':str(MANIFEST),'input_manifest_sha256':summary['manifest_sha256'],'no_simulink_run':True,'raw_inputs_read_only':True,'case_count':len(rows),'case_id_selection':a.case_ids or None,'limit':a.limit},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if len(rows)==4752 and len(good)==4752 and not errors:(target/'COMPLETE').write_text('All 4,752 cases read; raw-only detection and both window feature arms completed.\n',encoding='utf-8')
    print(json.dumps({'output':str(target),'summary':summary},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
