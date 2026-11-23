#!/usr/bin/env python3
"""Train on 0.1/1 ohm, select on training CV, final-test new 34 buses at 10 ohm."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.util,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
REPO=PACKAGE_ROOT.parents[1]
STEP2=PACKAGE_ROOT/'inputs'
MANIFEST=STEP2/'integrated_manifest_4752_final.csv'
PLACEMENT_FILES={n:REPO/f'data/reference/placements/wmu_count_comparison_{n}.csv' for n in ('ieee14','ieee30')}
PLACEMENTS={'ieee14':[2,6,9],'ieee30':[6,1,2]}
REPRESENTATIVE={'ieee14':{2,6,9,11,14},'ieee30':{1,6,10,24,30}}
NBUS={'ieee14':14,'ieee30':30};TRAIN_R={0.1,1.0};TEST_R=10.0;EPS=1e-12;N_SPLITS=5;SEED=42
from types import SimpleNamespace
mod=SimpleNamespace()
mod.C1_FEATURES=['pmu_V1','pmu_V2_over_V1','pmu_V0_over_V1','pmu_I1','pmu_I2_over_I1','pmu_I0_over_I1']
mod.C2_FEATURES=[]
for q in ('v','i'):
 for phase in 'ABC':mod.C2_FEATURES += [f'{q}_pre_rms_{phase}',f'{q}_event_rms_{phase}',f'{q}_post_rms_{phase}']
mod.C2_FEATURES += ['voltage_sag_ratio','voltage_phase_rms_mean','voltage_phase_rms_std','event_min_voltage','current_jump_ratio','current_phase_rms_mean','current_phase_rms_std','event_max_current','pre_to_event_voltage_change','pre_to_event_current_change','pre_to_post_voltage_change','pre_to_post_current_change']
mod.C2_FEATURES += ['V1','V2_over_V1','V0_over_V1','I1','I2_over_I1','I0_over_I1','sso_frequency_energy','lowfreq_5_45_energy','fundamental_magnitude','dominant_lowfreq_component']
mod.C3_FEATURES=['I_rel_sum','I_rel_max','V_rel_sum','V_rel_max']
mod.C4_FEATURES=[]
for q in ('v','i'):
 for phase in 'abc':mod.C4_FEATURES += [f'{q}_p2p_over_pre_rms_{phase}',f'{q}_derivative_rms_over_pre_rms_{phase}']
mod.ALL_FEATURES=list(dict.fromkeys(mod.C1_FEATURES+mod.C2_FEATURES+mod.C3_FEATURES+mod.C4_FEATURES))
FEATURE_SETS={
 'C1_modified_PMU_like':list(mod.C1_FEATURES),
 'C2_WMU_Baseline':list(mod.C2_FEATURES),
 'C3_Baseline_plus_NormVI':list(mod.C2_FEATURES)+list(mod.C3_FEATURES),
 'C4_C3_plus_time_domain':list(mod.C2_FEATURES)+list(mod.C3_FEATURES)+list(mod.C4_FEATURES),
}
MODE_FILES={'manifest_reference':'features_manifest_reference.csv.gz','detected':'features_detected.csv.gz'}
FORBIDDEN={'faulttype','faultbus','faultbuslabel','networkid','caseid','unifiedcaseid','backgroundname','ssofrequencyhz','ssomagnitudepct','faultresistanceohm','faultinceptionangledeg','eventstarttime','faultendtime','path','status','filename','sourcedataset','measuredbuslabel','windowmode','detectedonsets','truthonset'}
MODEL_ORDER=['ExtraTrees','RandomForest']

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read_csv(path):
 with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields):
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def validate_placements():
 hashes={}
 for net,path in PLACEMENT_FILES.items():
  rows=read_csv(path);sel=[r for r in rows if r.get('PlacementObjective')=='localization' and int(r.get('k',-1))==3]
  require(len(sel)==1,f'{net}: expected one localization k=3 placement row, got {len(sel)}')
  buses=[int(x) for x in sel[0]['SelectedWMUBuses'].split(';') if x]
  require(buses==PLACEMENTS[net],f'{net}: frozen 3-WMU placement mismatch {buses} != {PLACEMENTS[net]}')
  hashes[net]={'path':str(path),'sha256':sha256(path),'selected_buses':buses}
 return hashes
def load_manifest():
 rows=read_csv(MANIFEST);out={}
 require(len(rows)==4752,'integrated manifest must have 4,752 cases')
 for r in rows:
  uid=int(r['UnifiedCaseID']);require(uid not in out,f'duplicate case id {uid}')
  net=r['NetworkID'].lower();bus=int(r['FaultBus']);src=r['SourceDataset'];res=float(r['FaultResistanceOhm'])
  cohort='representative_10' if bus in REPRESENTATIVE[net] else 'new34'
  require((src=='baseline')==(cohort=='representative_10'),f'{net} bus {bus}: source/cohort mismatch')
  out[uid]={'UnifiedCaseID':uid,'NetworkID':net,'CaseID':int(r['SourceCaseID']),'SourceDataset':src,'Cohort':cohort,'FaultType':r['FaultType'],'FaultBusLabel':bus,'FaultResistanceOhm':res,'FaultInceptionAngleDeg':int(r['FaultInceptionAngleDeg']),'BackgroundName':r['BackgroundName'],'RawCSV_SHA256':r['ActualRawFileSHA256'],'TruthOnsetS':float(r['EventStartTime'])}
 require(sum(x['FaultResistanceOhm'] in TRAIN_R for x in out.values())==3168,'train-case count must be 3,168')
 require(sum(x['FaultResistanceOhm']==TEST_R and x['Cohort']=='new34' for x in out.values())==1224,'fresh new34 10-ohm test count must be 1,224')
 require(sum(x['FaultResistanceOhm']==TEST_R and x['Cohort']=='representative_10' for x in out.values())==360,'historically examined representative 10-ohm cases must be 360')
 return out
def load_runaways():
 p=STEP2/'onset_predictions.csv';out={}
 for r in read_csv(p):out[int(r['UnifiedCaseID'])]=str(r['RawOnlyVoltageRunawayFlag_gt_1e6']).lower()=='true'
 require(len(out)==4752,'Step 2 runaway flags must cover all cases')
 return out
def selected_normvi(case_features,buses):
 # Recompute NormVI after selecting the fixed three sensors; do not reuse full-network denominators.
 dv=[];di=[]
 for b in buses:
  r=case_features[b]
  vpre=float(np.mean([r[f'v_pre_rms_{p}'] for p in 'ABC']))
  ipre=float(np.mean([r[f'i_pre_rms_{p}'] for p in 'ABC']))
  dv.append(abs(float(r['voltage_phase_rms_mean'])-vpre))
  di.append(abs(float(r['current_phase_rms_mean'])-ipre))
 sv,si=sum(dv),sum(di);mv,mi=max(dv),max(di)
 for j,b in enumerate(buses):
  r=case_features[b]
  r['V_rel_sum']=dv[j]/(sv+EPS);r['V_rel_max']=dv[j]/(mv+EPS)
  r['I_rel_sum']=di[j]/(si+EPS);r['I_rel_max']=di[j]/(mi+EPS)
 return {'DeltaV':dv,'DeltaI':di,'SumV':sv,'SumI':si}
def load_feature_cases(meta,mode,include_ids):
 path=STEP2/MODE_FILES[mode];cases=defaultdict(dict);rows=0
 with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
  rd=csv.DictReader(f);headers=rd.fieldnames or []
  require(set(mod.ALL_FEATURES).issubset(headers),f'{mode}: missing Step 2 feature columns')
  for row in rd:
   uid=int(row['UnifiedCaseID'])
   if uid not in include_ids:continue
   m=meta[uid];net=m['NetworkID'];bus=int(row['MeasuredBusLabel'])
   if bus not in PLACEMENTS[net]:continue
   require(row['WindowMode']==mode,f'{mode}: window mismatch at {uid}')
   require(row['RawCSV_SHA256']==m['RawCSV_SHA256'],f'{mode}: raw hash mismatch at {uid}')
   require(row['FaultType']==m['FaultType'] and int(row['FaultBusLabel'])==m['FaultBusLabel'],f'{mode}: label mismatch at {uid}')
   feats={c:float(row[c]) for c in mod.ALL_FEATURES}
   require(all(math.isfinite(v) for v in feats.values()),f'{mode}: nonfinite feature {uid}/{bus}')
   require(bus not in cases[uid],f'{mode}: duplicate selected sensor feature row {uid}/{bus}')
   cases[uid][bus]=feats;rows+=1
 for uid,busrows in list(cases.items()):
  expected=set(PLACEMENTS[meta[uid]['NetworkID']])
  require(set(busrows)==expected,f'{mode}: selected sensor row incomplete for {uid}')
  selected_normvi(busrows,PLACEMENTS[meta[uid]['NetworkID']])
 return cases,rows
def vector(case_features,net,feature_set):
 features=FEATURE_SETS[feature_set]
 return np.asarray([case_features[b][f] for b in PLACEMENTS[net] for f in features],dtype=np.float64)
def labels(meta,ids,target):
 return np.asarray([meta[int(uid)]['FaultType'] if target=='FaultType' else meta[int(uid)]['FaultBusLabel'] for uid in ids],dtype=object if target=='FaultType' else np.int64)
def make_model(name):
 if name=='ExtraTrees':return ExtraTreesClassifier(n_estimators=120,random_state=SEED,n_jobs=-1,class_weight='balanced')
 return RandomForestClassifier(n_estimators=120,random_state=SEED,n_jobs=-1,class_weight='balanced_subsample')
def metrics(y,pred,classes):
 y=np.asarray(y);pred=np.asarray(pred)
 return {'Accuracy':float(accuracy_score(y,pred)),'MacroF1':float(f1_score(y,pred,labels=list(classes),average='macro',zero_division=0))}
def classes_for(net,target):
 return ['SLG','LL','LLG','ThreePhase'] if target=='FaultType' else list(range(1,NBUS[net]+1))
def mode_case_order(meta,case_features,net,mode,resistance_set,cohort=None,exclude_runaways=None,runaways=None):
 ids=sorted(u for u,m in meta.items() if m['NetworkID']==net and m['FaultResistanceOhm'] in resistance_set and (cohort is None or m['Cohort']==cohort) and (exclude_runaways is None or not (exclude_runaways and runaways[u])) and u in case_features)
 return ids
def train_cv(meta,feature_cases,mode,runaways,exclude_train_runaways=False):
 cv_rows=[];fold_rows=[];cv_predictions=[];selection=[];fold_maps={}
 for net in ('ieee14','ieee30'):
  ids=mode_case_order(meta,feature_cases,net,mode,TRAIN_R,exclude_runaways=exclude_train_runaways,runaways=runaways)
  require(ids,f'{net}/{mode}: no training cases')
  for target in ('FaultType','FaultBusLabel'):
   y=labels(meta,ids,target);groups=np.asarray(ids,dtype=np.int64);classes=classes_for(net,target)
   splitter=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
   folds=list(splitter.split(np.zeros((len(ids),1)),y,groups));assign=np.full(len(ids),-1,dtype=np.int8)
   for fold,(tr,te) in enumerate(folds,1):
    require(not set(groups[tr])&set(groups[te]),'CaseID overlap in train-only CV')
    assign[te]=fold
    fold_rows.extend({'NetworkID':net,'Target':target,'WindowMode':mode,'UnifiedCaseID':int(ids[i]),'Fold':fold,'TruthLabel':str(y[i])} for i in te)
   require(np.all(assign>0),'missing CV fold assignment')
   fold_maps[(net,target,mode)]=(ids,assign)
   candidate_rows=[]
   for fs,feat_names in FEATURE_SETS.items():
    X=np.vstack([vector(feature_cases[u],net,fs) for u in ids])
    require(X.shape[1]==len(feat_names)*3,f'wrong feature width {net}/{fs}')
    for model_name in MODEL_ORDER:
     fold_metrics=[]
     for fold,(tr,te) in enumerate(folds,1):
      clf=make_model(model_name);clf.fit(X[tr],y[tr]);p=clf.predict(X[te]);m=metrics(y[te],p,classes)
      fold_metrics.append(m)
      for ix,pr in zip(te,p):
       cv_predictions.append({'NetworkID':net,'Target':target,'WindowMode':mode,'FeatureSet':fs,'Model':model_name,'Fold':fold,'UnifiedCaseID':int(ids[ix]),'TruthLabel':str(y[ix]),'PredictedLabel':str(pr),'Correct':bool(pr==y[ix])})
     row={'NetworkID':net,'Target':target,'WindowMode':mode,'FeatureSet':fs,'Model':model_name,'TrainCases':len(ids),'FeatureColumns':len(feat_names)*3,'MacroF1_FoldMean':float(np.mean([x['MacroF1'] for x in fold_metrics])),'MacroF1_FoldStd':float(np.std([x['MacroF1'] for x in fold_metrics])),'Accuracy_FoldMean':float(np.mean([x['Accuracy'] for x in fold_metrics])),'Accuracy_FoldStd':float(np.std([x['Accuracy'] for x in fold_metrics])),'FoldCount':N_SPLITS,'TrainingRunawayExcluded':exclude_train_runaways}
     candidate_rows.append(row);cv_rows.append(row)
   ranked=sorted(candidate_rows,key=lambda r:(-r['MacroF1_FoldMean'],-r['Accuracy_FoldMean'],r['FeatureColumns'],MODEL_ORDER.index(r['Model']),list(FEATURE_SETS).index(r['FeatureSet'])))
   chosen=ranked[0]
   for rank,r in enumerate(ranked,1):
    r['TrainCVRank']=rank;r['SelectedByTrainCV']=rank==1
   selection.append({'NetworkID':net,'Target':target,'WindowMode':mode,'SelectedModel':chosen['Model'],'SelectedFeatureSet':chosen['FeatureSet'],'SelectionMacroF1':chosen['MacroF1_FoldMean'],'SelectionAccuracy':chosen['Accuracy_FoldMean'],'FeatureColumns':chosen['FeatureColumns'],'SelectionRule':'max fold-mean MacroF1, then Accuracy; ties prefer fewer features, ExtraTrees, then C1-C4 order','TrainingRunawayExcluded':exclude_train_runaways})
 return cv_rows,fold_rows,cv_predictions,selection,fold_maps
def predict_test(meta,train_cases,test_cases,mode,selection,runaways):
 metric_rows=[];pred_rows=[];conf_rows=[];perbus_rows=[];pertype_rows=[]
 for net in ('ieee14','ieee30'):
  train_ids=mode_case_order(meta,train_cases,net,mode,TRAIN_R,runaways=runaways)
  test_ids=mode_case_order(meta,test_cases,net,mode,{TEST_R},cohort='new34',runaways=runaways)
  require(train_ids,'empty train set');train_by_target={t:labels(meta,train_ids,t) for t in ('FaultType','FaultBusLabel')}
  cohort_total=sum(1 for m in meta.values() if m['NetworkID']==net and m['FaultResistanceOhm']==TEST_R and m['Cohort']=='new34')
  for target in ('FaultType','FaultBusLabel'):
   classes=classes_for(net,target);ytrain=train_by_target[target]
   for fs,feat_names in FEATURE_SETS.items():
    Xtrain=np.vstack([vector(train_cases[u],net,fs) for u in train_ids])
    for model_name in MODEL_ORDER:
     clf=make_model(model_name);clf.fit(Xtrain,ytrain)
     valid_ids=[u for u in test_ids if u in test_cases]
     Xtest=np.vstack([vector(test_cases[u],net,fs) for u in valid_ids]) if valid_ids else np.empty((0,len(feat_names)*3))
     pred=clf.predict(Xtest) if valid_ids else np.asarray([])
     chosen=next(x for x in selection if x['NetworkID']==net and x['Target']==target and x['WindowMode']==mode)
     selected=(chosen['SelectedModel']==model_name and chosen['SelectedFeatureSet']==fs)
     valid_pred={u:p for u,p in zip(valid_ids,pred)}
     for uid in sorted(u for u,m in meta.items() if m['NetworkID']==net and m['FaultResistanceOhm']==TEST_R and m['Cohort']=='new34'):
      truth=meta[uid]['FaultType'] if target=='FaultType' else meta[uid]['FaultBusLabel'];has=uid in valid_pred;pr=valid_pred.get(uid,'')
      pred_rows.append({'NetworkID':net,'Cohort':'new34','Target':target,'WindowMode':mode,'Model':model_name,'FeatureSet':fs,'SelectedByTrainCV':selected,'UnifiedCaseID':uid,'FaultBusLabel':meta[uid]['FaultBusLabel'],'FaultType':meta[uid]['FaultType'],'FaultResistanceOhm':meta[uid]['FaultResistanceOhm'],'BackgroundName':meta[uid]['BackgroundName'],'FaultInceptionAngleDeg':meta[uid]['FaultInceptionAngleDeg'],'RawOnlyVoltageRunawayFlag_gt_1e6':runaways[uid],'ValidFeatureWindow':has,'TruthLabel':str(truth),'PredictedLabel':str(pr) if has else '', 'Correct':bool(has and pr==truth)})
     if valid_ids:
      m=metrics([meta[u]['FaultType'] if target=='FaultType' else meta[u]['FaultBusLabel'] for u in valid_ids],pred,classes)
      ncorrect=sum(bool(p==(meta[u]['FaultType'] if target=='FaultType' else meta[u]['FaultBusLabel'])) for u,p in zip(valid_ids,pred))
     else:m={'Accuracy':None,'MacroF1':None};ncorrect=0
     cov=len(valid_ids)/cohort_total if cohort_total else 0
     metric_rows.append({'NetworkID':net,'Cohort':'new34','Target':target,'WindowMode':mode,'Model':model_name,'FeatureSet':fs,'SelectedByTrainCV':selected,'TrainCases':len(train_ids),'TestCasesTotal':cohort_total,'TestCasesWithValidWindow':len(valid_ids),'WindowCoverage':cov,'ConditionalAccuracy':m['Accuracy'],'ConditionalMacroF1':m['MacroF1'],'EndToEndCorrectRate':ncorrect/cohort_total if cohort_total else None,'CorrectValidPredictions':ncorrect,'FeatureColumns':len(feat_names)*3})
     if valid_ids:
      yt=[meta[u]['FaultType'] if target=='FaultType' else meta[u]['FaultBusLabel'] for u in valid_ids]
      cm=confusion_matrix(yt,pred,labels=classes)
      for i,truth_label in enumerate(classes):
       for j,pred_label in enumerate(classes):
        conf_rows.append({'NetworkID':net,'Cohort':'new34','Target':target,'WindowMode':mode,'Model':model_name,'FeatureSet':fs,'TruthLabel':str(truth_label),'PredictedLabel':str(pred_label),'Count':int(cm[i,j])})
      if target=='FaultBusLabel':
       for bus in sorted({meta[u]['FaultBusLabel'] for u in valid_ids}):
        ix=[i for i,u in enumerate(valid_ids) if meta[u]['FaultBusLabel']==bus]
        bcorrect=sum(int(pred[i]==bus) for i in ix)
        perbus_rows.append({'NetworkID':net,'Cohort':'new34','WindowMode':mode,'Model':model_name,'FeatureSet':fs,'SelectedByTrainCV':selected,'FaultBusLabel':bus,'TestCasesTotal':sum(1 for m in meta.values() if m['NetworkID']==net and m['FaultResistanceOhm']==TEST_R and m['Cohort']=='new34' and m['FaultBusLabel']==bus),'ValidWindowCases':len(ix),'Correct':bcorrect,'ConditionalExactBusAccuracy':bcorrect/len(ix) if ix else None})
      else:
       for ft in classes:
        ix=[i for i,u in enumerate(valid_ids) if meta[u]['FaultType']==ft]
        fcorrect=sum(int(pred[i]==ft) for i in ix)
        pertype_rows.append({'NetworkID':net,'Cohort':'new34','WindowMode':mode,'Model':model_name,'FeatureSet':fs,'SelectedByTrainCV':selected,'FaultType':ft,'TestCasesTotal':sum(1 for m in meta.values() if m['NetworkID']==net and m['FaultResistanceOhm']==TEST_R and m['Cohort']=='new34' and m['FaultType']==ft),'ValidWindowCases':len(ix),'Correct':fcorrect,'ConditionalAccuracy':fcorrect/len(ix) if ix else None})
 return metric_rows,pred_rows,conf_rows,perbus_rows,pertype_rows
def sensitivity_selected(meta,feature_cases_by_mode,selection_by_mode,runaways):
 rows=[]
 for mode,features in feature_cases_by_mode.items():
  for sel in selection_by_mode[mode]:
   net,target,fs,model=sel['NetworkID'],sel['Target'],sel['SelectedFeatureSet'],sel['SelectedModel']
   train_ids=mode_case_order(meta,features,net,mode,TRAIN_R,exclude_runaways=True,runaways=runaways)
   test_ids=mode_case_order(meta,features,net,mode,{TEST_R},cohort='new34',runaways=runaways)
   y=labels(meta,train_ids,target);X=np.vstack([vector(features[u],net,fs) for u in train_ids]);clf=make_model(model);clf.fit(X,y)
   valid=[u for u in test_ids if u in features]
   pred=clf.predict(np.vstack([vector(features[u],net,fs) for u in valid])) if valid else []
   classes=classes_for(net,target);m=metrics([meta[u]['FaultType'] if target=='FaultType' else meta[u]['FaultBusLabel'] for u in valid],pred,classes) if valid else {'Accuracy':None,'MacroF1':None}
   total=sum(1 for z in meta.values() if z['NetworkID']==net and z['FaultResistanceOhm']==TEST_R and z['Cohort']=='new34')
   correct=sum(bool(pr==(meta[u]['FaultType'] if target=='FaultType' else meta[u]['FaultBusLabel'])) for u,pr in zip(valid,pred))
   rows.append({'NetworkID':net,'Target':target,'WindowMode':mode,'SelectedModel':model,'SelectedFeatureSet':fs,'TrainingCasesAfterRunawayExclusion':len(train_ids),'RunawayTrainCasesExcluded':sum(runaways[u] for u in mode_case_order(meta,features,net,mode,TRAIN_R,runaways=runaways)),'TestCasesTotal':total,'ValidTestWindows':len(valid),'WindowCoverage':len(valid)/total,'ConditionalAccuracy':m['Accuracy'],'ConditionalMacroF1':m['MacroF1'],'EndToEndCorrectRate':correct/total,'SelectionWasOriginallyOnFullTrainingSet':True})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='outputs/reproduction_v1');args=ap.parse_args()
 out=(PACKAGE_ROOT/args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
 if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
 require(json.loads((STEP2/'STEP2_VERIFICATION_FINAL.json').read_text())['ml_evaluation']['status']=='PASS','Step 2 final verification not PASS')
 placements_meta=validate_placements();meta=load_manifest();runaways=load_runaways()
 train_ids={u for u,m in meta.items() if m['FaultResistanceOhm'] in TRAIN_R}
 new_test_ids={u for u,m in meta.items() if m['FaultResistanceOhm']==TEST_R and m['Cohort']=='new34'}
 include=train_ids|new_test_ids
 out.mkdir(parents=True);(out/'models').mkdir();(out/'tables').mkdir();(out/'predictions').mkdir();(out/'splits').mkdir()
 feature_cases_by_mode={};feature_rows={}
 for mode in MODE_FILES:
  feature_cases_by_mode[mode],feature_rows[mode]=load_feature_cases(meta,mode,include)
  expected_train={u for u in train_ids}
  missing_train=expected_train-set(feature_cases_by_mode[mode])
  require(not missing_train,f'{mode}: train window missing for {len(missing_train)} cases')
  # Keep only training plus previously unseen new-bus 10-ohm cases; representative 10-ohm holdout remains unopened.
  require(not any(meta[u]['FaultResistanceOhm']==TEST_R and meta[u]['Cohort']=='representative_10' for u in feature_cases_by_mode[mode]),'representative 10-ohm holdout was unexpectedly loaded')

 all_cv=[];all_folds=[];all_cv_preds=[];all_selection=[];foldmaps={}
 for mode,features in feature_cases_by_mode.items():
  cv,folds,preds,selection,fmap=train_cv(meta,features,mode,runaways,False)
  all_cv+=cv;all_folds+=folds;all_cv_preds+=preds;all_selection+=selection;foldmaps[mode]=fmap
 write_csv(out/'tables/train_only_cv.csv',all_cv,['NetworkID','Target','WindowMode','FeatureSet','Model','TrainCases','FeatureColumns','MacroF1_FoldMean','MacroF1_FoldStd','Accuracy_FoldMean','Accuracy_FoldStd','FoldCount','TrainingRunawayExcluded','TrainCVRank','SelectedByTrainCV'])
 write_csv(out/'splits/train_cv_fold_assignments.csv',all_folds,['NetworkID','Target','WindowMode','UnifiedCaseID','Fold','TruthLabel'])
 write_csv(out/'predictions/train_cv_oof_predictions.csv',all_cv_preds,['NetworkID','Target','WindowMode','FeatureSet','Model','Fold','UnifiedCaseID','TruthLabel','PredictedLabel','Correct'])
 write_csv(out/'tables/model_selection.csv',all_selection,['NetworkID','Target','WindowMode','SelectedModel','SelectedFeatureSet','SelectionMacroF1','SelectionAccuracy','FeatureColumns','SelectionRule','TrainingRunawayExcluded'])
 all_metrics=[];all_preds=[];all_conf=[];all_perbus=[];all_pertype=[]
 for mode,features in feature_cases_by_mode.items():
  selected=[r for r in all_selection if r['WindowMode']==mode]
  metrics_rows,pred_rows,conf_rows,perbus,pertype=predict_test(meta,features,features,mode,selected,runaways)
  all_metrics+=metrics_rows;all_preds+=pred_rows;all_conf+=conf_rows;all_perbus+=perbus;all_pertype+=pertype
 write_csv(out/'tables/new34_10ohm_test_metrics.csv',all_metrics,['NetworkID','Cohort','Target','WindowMode','Model','FeatureSet','SelectedByTrainCV','TrainCases','TestCasesTotal','TestCasesWithValidWindow','WindowCoverage','ConditionalAccuracy','ConditionalMacroF1','EndToEndCorrectRate','CorrectValidPredictions','FeatureColumns'])
 write_csv(out/'predictions/new34_10ohm_test_predictions.csv',all_preds,['NetworkID','Cohort','Target','WindowMode','Model','FeatureSet','SelectedByTrainCV','UnifiedCaseID','FaultBusLabel','FaultType','FaultResistanceOhm','BackgroundName','FaultInceptionAngleDeg','RawOnlyVoltageRunawayFlag_gt_1e6','ValidFeatureWindow','TruthLabel','PredictedLabel','Correct'])
 write_csv(out/'tables/confusion_matrices_new34_10ohm.csv',all_conf,['NetworkID','Cohort','Target','WindowMode','Model','FeatureSet','TruthLabel','PredictedLabel','Count'])
 write_csv(out/'tables/per_bus_localization_new34_10ohm.csv',all_perbus,['NetworkID','Cohort','WindowMode','Model','FeatureSet','SelectedByTrainCV','FaultBusLabel','TestCasesTotal','ValidWindowCases','Correct','ConditionalExactBusAccuracy'])
 write_csv(out/'tables/per_fault_type_new34_10ohm.csv',all_pertype,['NetworkID','Cohort','WindowMode','Model','FeatureSet','SelectedByTrainCV','FaultType','TestCasesTotal','ValidWindowCases','Correct','ConditionalAccuracy'])
 sensitivity=[]
 for mode,features in feature_cases_by_mode.items():
  selected=[r for r in all_selection if r['WindowMode']==mode]
  sensitivity+=sensitivity_selected(meta,{mode:features},{mode:selected},runaways)
 write_csv(out/'tables/runaway_training_sensitivity.csv',sensitivity,['NetworkID','Target','WindowMode','SelectedModel','SelectedFeatureSet','TrainingCasesAfterRunawayExclusion','RunawayTrainCasesExcluded','TestCasesTotal','ValidTestWindows','WindowCoverage','ConditionalAccuracy','ConditionalMacroF1','EndToEndCorrectRate','SelectionWasOriginallyOnFullTrainingSet'])
 leakage=[]
 for net in ('ieee14','ieee30'):
  for mode in MODE_FILES:
   for fs,feats in FEATURE_SETS.items():
    bad=[f for f in feats if any(term in f.lower() for term in FORBIDDEN)]
    leakage.append({'NetworkID':net,'WindowMode':mode,'FeatureSet':fs,'SelectedWMUBuses':';'.join(map(str,PLACEMENTS[net])),'FeatureCountPerWMU':len(feats),'InputColumnCount':3*len(feats),'ForbiddenFeatureNames':';'.join(bad),'Status':'PASS' if not bad else 'FAIL','NormVIDenominator':'selected three WMUs only' if fs in ('C3_Baseline_plus_NormVI','C4_C3_plus_time_domain') else 'not applicable'})
 require(all(r['Status']=='PASS' for r in leakage),'feature leakage audit failed')
 write_csv(out/'tables/feature_leakage_audit.csv',leakage,['NetworkID','WindowMode','FeatureSet','SelectedWMUBuses','FeatureCountPerWMU','InputColumnCount','ForbiddenFeatureNames','Status','NormVIDenominator'])
 inventory=[]
 for net in ('ieee14','ieee30'):
  for fs,feats in FEATURE_SETS.items():
   for bus in PLACEMENTS[net]:
    for f in feats:inventory.append({'NetworkID':net,'FeatureSet':fs,'ModelFeatureName':f'Bus{bus:02d}__{f}','RawFeatureName':f,'SelectedWMUBuses':';'.join(map(str,PLACEMENTS[net]))})
 write_csv(out/'tables/model_feature_columns.csv',inventory,['NetworkID','FeatureSet','ModelFeatureName','RawFeatureName','SelectedWMUBuses'])
 # Audit exact cohort counts and frozen resistance split; representative 10-ohm cases are intentionally not evaluated here.
 source_counts=Counter((m['NetworkID'],m['Cohort'],m['FaultResistanceOhm']) for m in meta.values())
 expected={'ieee14':{'baseline_train':360,'new_train':648,'new_test':324},'ieee30':{'baseline_train':360,'new_train':1800,'new_test':900}}
 for net in ('ieee14','ieee30'):
  require(source_counts[(net,'representative_10',0.1)]+source_counts[(net,'representative_10',1.0)]==expected[net]['baseline_train'],f'{net} representative train count mismatch')
  require(source_counts[(net,'new34',0.1)]+source_counts[(net,'new34',1.0)]==expected[net]['new_train'],f'{net} new34 train count mismatch')
  require(source_counts[(net,'new34',10.0)]==expected[net]['new_test'],f'{net} new34 test count mismatch')
 runmeta={'status':'COMPLETE','scope':'10-ohm final test on new34 fault buses only; representative 10-ohm model features/predictions excluded (manifest metadata retained for cohort audit)','step2_input_root':str(STEP2),'integrated_manifest_sha256':sha256(MANIFEST),'step2_detected_features_sha256':sha256(STEP2/'features_detected.csv.gz'),'step2_manifest_reference_features_sha256':sha256(STEP2/'features_manifest_reference.csv.gz'),'step2_onset_predictions_sha256':sha256(STEP2/'onset_predictions.csv'),'step2_extractor_sha256':sha256(PACKAGE_ROOT/'source_snapshot/extract_step2_waveform_features.py'),'step2_detector_sha256':sha256(PACKAGE_ROOT/'source_snapshot/waveform.py'),'step3_evaluator_sha256':sha256(Path(__file__)),'placement_sources':placements_meta,'fixed_placements':PLACEMENTS,'representative_fault_buses':{n:sorted(v) for n,v in REPRESENTATIVE.items()},'new34_fault_buses':{'ieee14':[b for b in range(1,15) if b not in REPRESENTATIVE['ieee14']],'ieee30':[b for b in range(1,31) if b not in REPRESENTATIVE['ieee30']]},'train_resistances_ohm':[0.1,1.0],'final_test_resistance_ohm':10.0,'training_cases_by_network':{n:sum(1 for m in meta.values() if m['NetworkID']==n and m['FaultResistanceOhm'] in TRAIN_R) for n in ('ieee14','ieee30')},'new34_10ohm_test_cases_by_network':{n:sum(1 for m in meta.values() if m['NetworkID']==n and m['FaultResistanceOhm']==10.0 and m['Cohort']=='new34') for n in ('ieee14','ieee30')},'representative_10ohm_test_cases_not_scored':360,'feature_sets_per_sensor':{k:len(v) for k,v in FEATURE_SETS.items()},'model_hyperparameters':{'ExtraTrees':'n_estimators=120, random_state=42, n_jobs=-1, class_weight=balanced','RandomForest':'n_estimators=120, random_state=42, n_jobs=-1, class_weight=balanced_subsample'},'model_selection':'5-fold StratifiedGroupKFold on 0.1/1-ohm training cases only; groups=UnifiedCaseID; compare 2 estimators × C1-C4; choose max mean MacroF1 then Accuracy; ties prefer fewer features, ExtraTrees, then C1-C4 order. No 10-ohm labels/features enter selection.','new34_test_feature_windows':{'manifest_reference':'all manifest-timed reference windows','detected':'Step 2 raw-only D1 detected windows; valid-window coverage is reported, and missing windows are incorrect in end-to-end correct rate.'},'selected_only_normvi':'C3/C4 NormVI denominators are recomputed per case over the selected three WMUs only; no full-network or test-population denominator.','training_runaway_sensitivity':'Primary selection/training retains flagged 18 records. Separate selected-model sensitivity refits after excluding flagged training records; does not retune model/feature on 10-ohm test.','no_physical_topology_claim':True,'tests_expected':['grouped case split disjointness','selected-only NormVI sum/max invariant','feature-set membership and no label metadata inputs','test cohort exclusion of representative 10-ohm holdout']}
 (out/'run_metadata.json').write_text(json.dumps(runmeta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({'output':str(out),'train_cases':runmeta['training_cases_by_network'],'new34_test':runmeta['new34_10ohm_test_cases_by_network'],'cv_candidate_rows':len(all_cv),'cv_oof_rows':len(all_cv_preds),'final_metrics_rows':len(all_metrics),'final_prediction_rows':len(all_preds),'selected_candidates':all_selection,'runaway_sensitivity_rows':len(sensitivity),'feature_rows':feature_rows},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
