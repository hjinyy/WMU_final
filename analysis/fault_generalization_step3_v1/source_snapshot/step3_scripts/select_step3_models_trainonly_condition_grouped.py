#!/usr/bin/env python3
"""Train-only CV sensitivity grouping paired resistance cases by operating condition."""
import csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO=Path('/home/hy/WMU_project')
MAIN=REPO/'results/fault_generalization_step3_v1/scripts/evaluate_step3_unseen10_fixed3wmu.py'
spec=importlib.util.spec_from_file_location('step3_main',MAIN)
if spec is None or spec.loader is None:raise RuntimeError('cannot load Step3 helper')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

def main():
 out=REPO/'results/fault_generalization_step3_v1/train_only_condition_grouped_v1'
 if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
 out.mkdir(parents=True);(out/'tables').mkdir();(out/'splits').mkdir()
 meta=base.load_manifest();train_ids={u for u,m in meta.items() if m['FaultResistanceOhm'] in base.TRAIN_R}
 results=[];assignments=[]
 for mode in base.MODE_FILES:
  data,feature_rows=base.load_feature_cases(meta,mode,train_ids)
  if set(data)!=train_ids:raise ValueError(f'{mode}: train data missing {len(train_ids-set(data))} cases')
  for net in ('ieee14','ieee30'):
   ids=sorted(u for u in train_ids if meta[u]['NetworkID']==net)
   group_ids=[f"bus={meta[u]['FaultBusLabel']}|type={meta[u]['FaultType']}|angle={meta[u]['FaultInceptionAngleDeg']}|background={meta[u]['BackgroundName']}" for u in ids]
   if any(len({meta[u]['FaultResistanceOhm'] for u,g in zip(ids,group_ids) if g==key})!=2 for key in set(group_ids)):
    raise ValueError(f'{net}: condition group does not contain both development resistances')
   for target in ('FaultType','FaultBusLabel'):
    y=base.labels(meta,ids,target);classes=base.classes_for(net,target)
    splitter=StratifiedGroupKFold(n_splits=base.N_SPLITS,shuffle=True,random_state=base.SEED)
    folds=list(splitter.split(np.zeros((len(ids),1)),y,np.asarray(group_ids,dtype=object)))
    fold_assign=np.full(len(ids),-1,dtype=int)
    for fold,(tr,te) in enumerate(folds,1):
     if set(np.asarray(group_ids,dtype=object)[tr])&set(np.asarray(group_ids,dtype=object)[te]):raise ValueError('condition group overlap across CV folds')
     fold_assign[te]=fold
     for i in te:assignments.append({'NetworkID':net,'Target':target,'WindowMode':mode,'UnifiedCaseID':ids[i],'ConditionGroupID':group_ids[i],'FaultResistanceOhm':meta[ids[i]]['FaultResistanceOhm'],'Fold':fold})
    candidates=[]
    for fs,feat_names in base.FEATURE_SETS.items():
     X=np.vstack([base.vector(data[u],net,fs) for u in ids])
     for model in base.MODEL_ORDER:
      fm=[]
      for fold,(tr,te) in enumerate(folds,1):
       clf=base.make_model(model);clf.fit(X[tr],y[tr]);pred=clf.predict(X[te]);fm.append(base.metrics(y[te],pred,classes))
      r={'NetworkID':net,'Target':target,'WindowMode':mode,'FeatureSet':fs,'Model':model,'TrainCases':len(ids),'ConditionGroups':len(set(group_ids)),'MacroF1_FoldMean':float(np.mean([m['MacroF1'] for m in fm])),'MacroF1_FoldStd':float(np.std([m['MacroF1'] for m in fm])),'Accuracy_FoldMean':float(np.mean([m['Accuracy'] for m in fm])),'Accuracy_FoldStd':float(np.std([m['Accuracy'] for m in fm])),'FoldCount':base.N_SPLITS,'GroupDefinition':'Network + FaultBus + FaultType + FaultInceptionAngleDeg + BackgroundName; resistance deliberately excluded'}
      candidates.append(r)
    ranked=sorted(candidates,key=lambda r:(-r['MacroF1_FoldMean'],-r['Accuracy_FoldMean'],len(base.FEATURE_SETS[r['FeatureSet']])*3,base.MODEL_ORDER.index(r['Model']),list(base.FEATURE_SETS).index(r['FeatureSet'])))
    for rank,r in enumerate(ranked,1):r['TrainCVRank']=rank;r['SelectedByConditionGroupedCV']=rank==1;results.append(r)
 base.write_csv(out/'tables/train_only_condition_grouped_cv.csv',results,['NetworkID','Target','WindowMode','FeatureSet','Model','TrainCases','ConditionGroups','MacroF1_FoldMean','MacroF1_FoldStd','Accuracy_FoldMean','Accuracy_FoldStd','FoldCount','GroupDefinition','TrainCVRank','SelectedByConditionGroupedCV'])
 base.write_csv(out/'splits/train_only_condition_grouped_folds.csv',assignments,['NetworkID','Target','WindowMode','UnifiedCaseID','ConditionGroupID','FaultResistanceOhm','Fold'])
 selections=[]
 for r in results:
  if r['TrainCVRank']==1:selections.append({k:r[k] for k in ['NetworkID','Target','WindowMode','Model','FeatureSet','MacroF1_FoldMean','Accuracy_FoldMean','ConditionGroups','GroupDefinition']})
 (out/'run_metadata.json').write_text(json.dumps({'status':'COMPLETE','scope':'train-only CV sensitivity; no 10-ohm test features or labels used','training_resistances_ohm':[0.1,1.0],'candidate_models':['ExtraTrees','RandomForest'],'feature_sets_per_sensor':{k:len(v) for k,v in base.FEATURE_SETS.items()},'folds':base.N_SPLITS,'grouping':'Network + FaultBus + FaultType + FaultInceptionAngleDeg + BackgroundName; pair the 0.1 and 1 ohm cases for each factorial condition in the same fold','selection_rule':'max fold-mean MacroF1, then Accuracy; ties prefer fewer features, ExtraTrees, then C1-C4 order','selection':selections,'feature_rows_by_window_mode':feature_rows},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({'output':str(out),'candidate_rows':len(results),'fold_assignment_rows':len(assignments),'selection':selections,'feature_rows':feature_rows},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
