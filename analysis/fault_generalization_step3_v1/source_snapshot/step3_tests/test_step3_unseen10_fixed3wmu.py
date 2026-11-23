import importlib.util
import unittest
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT=Path(__file__).parents[1]/'scripts/evaluate_step3_unseen10_fixed3wmu.py'
spec=importlib.util.spec_from_file_location('step3_eval',SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError('could not load Step 3 evaluator')
step3=importlib.util.module_from_spec(spec);spec.loader.exec_module(step3)

class Step3PipelineTests(unittest.TestCase):
    def test_fixed_three_wmu_matches_existing_summary(self):
        evidence=step3.validate_placements()
        self.assertEqual(evidence['ieee14']['selected_buses'],[2,6,9])
        self.assertEqual(evidence['ieee30']['selected_buses'],[6,1,2])

    def test_cohort_and_resistance_split_counts(self):
        meta=step3.load_manifest()
        train={u for u,m in meta.items() if m['FaultResistanceOhm'] in step3.TRAIN_R}
        test={u for u,m in meta.items() if m['FaultResistanceOhm']==step3.TEST_R and m['Cohort']=='new34'}
        old10={u for u,m in meta.items() if m['FaultResistanceOhm']==step3.TEST_R and m['Cohort']=='representative_10'}
        self.assertEqual((len(train),len(test),len(old10)),(3168,1224,360))
        self.assertFalse(train & test)
        self.assertTrue(all(meta[u]['SourceDataset']=='new' for u in test))
        self.assertTrue(all(meta[u]['FaultResistanceOhm']<step3.TEST_R for u in train))

    def test_selected_only_normvi_sums_and_maxima(self):
        buses=[2,6,9];rows={}
        for i,b in enumerate(buses,1):
            rows[b]={k:0.0 for k in step3.mod.ALL_FEATURES}
            rows[b].update({'v_pre_rms_A':1.0,'v_pre_rms_B':1.0,'v_pre_rms_C':1.0,
                            'i_pre_rms_A':1.0,'i_pre_rms_B':1.0,'i_pre_rms_C':1.0,
                            'voltage_phase_rms_mean':1.0+i,'current_phase_rms_mean':1.0+2*i})
        step3.selected_normvi(rows,buses)
        self.assertTrue(np.isclose(sum(rows[b]['V_rel_sum'] for b in buses),1.0))
        self.assertTrue(np.isclose(sum(rows[b]['I_rel_sum'] for b in buses),1.0))
        self.assertTrue(np.isclose(max(rows[b]['V_rel_max'] for b in buses),1.0))
        self.assertTrue(np.isclose(max(rows[b]['I_rel_max'] for b in buses),1.0))

    def test_feature_sets_exclude_labels_and_have_expected_widths(self):
        self.assertEqual([len(step3.FEATURE_SETS[k]) for k in step3.FEATURE_SETS],[6,40,44,56])
        for features in step3.FEATURE_SETS.values():
            self.assertFalse(any(any(token in f.lower() for token in step3.FORBIDDEN) for f in features))

    def test_grouped_training_folds_are_disjoint_and_cover_cases(self):
        groups=np.arange(80);y=np.tile([0,1,2,3],20)
        split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=step3.SEED)
        seen=[]
        for train,test in split.split(np.zeros((len(y),1)),y,groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen.extend(test.tolist())
        self.assertEqual(sorted(seen),list(range(len(y))))

if __name__=='__main__':
    unittest.main()
