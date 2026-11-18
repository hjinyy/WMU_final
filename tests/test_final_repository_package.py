from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_final_repository_contains_only_expected_figure_outputs():
 a=ROOT/'results/paper_figures_resistance_generalization_v1'
 b=ROOT/'results/paper_figures_resistance_generalization_v1_fig04_sso_audited_replacement'
 c=ROOT/'results/fig04_sso_pre_fault_persistence_audit_v1'
 assert len(list((a/'png').glob('*.png')))==8
 assert len(list((a/'pdf').glob('*.pdf')))==8
 assert len(list((a/'captions').glob('*.md')))==8
 assert (b/'png/fig04_sso_audited_replacement.png').is_file()
 assert (c/'png/fig04_pre_fault_persistence.png').is_file()
def test_only_required_selected_waveform_channels_are_packaged():
 p=ROOT/'data/source_corpus/raw_csv'
 fs=list(p.glob('*.csv'));assert len(fs)==6
 prov=pd.read_csv(ROOT/'data/source_case_provenance.csv');assert len(prov)==6
 assert prov.OriginalSHA256.str.fullmatch(r'[0-9a-f]{64}').all()
 for f in fs:
  d=pd.read_csv(f,nrows=1)
  assert 'Time' in d.columns
  assert set(d.columns).issubset({'Time'}|{f'{ph}_{b}' for ph in ['Va','Ia'] for b in range(1,31)})
def test_local_reproduction_inputs_are_present():
 assert (ROOT/'data/source_corpus/features/ieee14_fault_generalization_features.csv.gz').is_file()
 assert (ROOT/'data/source_corpus/features/ieee30_fault_generalization_features.csv.gz').is_file()
 assert (ROOT/'results/high_resistance_ml_baseline_audit_v1/predictions/resistance_holdout_predictions.csv').is_file()
 assert (ROOT/'results/high_resistance_fault_type_improvement_v1/predictions/heldout_10ohm_predictions.csv').is_file()
