from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_repository_has_separate_data_code_outputs_and_test_roles():
 assert (ROOT/'data/frozen_inputs').is_dir()
 assert (ROOT/'src/wmu_final/features.py').is_file()
 assert (ROOT/'scripts/render').is_dir()
 assert (ROOT/'scripts/audit').is_dir()
 assert (ROOT/'outputs').is_dir()
 assert not (ROOT/'results').exists()
 assert not (ROOT/'scripts/run_unseen_resistance_spatial_generalization_v2.py').exists()
def test_final_repository_contains_only_expected_figure_outputs():
 a=ROOT/'outputs/paper_figures_v1'
 b=ROOT/'outputs/fig04_sso_audited_replacement_v1'
 c=ROOT/'outputs/fig04_pre_fault_persistence_audit_v1'
 assert len(list((a/'png').glob('*.png')))==8
 assert len(list((a/'pdf').glob('*.pdf')))==8
 assert len(list((a/'captions').glob('*.md')))==8
 assert (b/'png/fig04_sso_audited_replacement.png').is_file()
 assert (c/'png/fig04_pre_fault_persistence.png').is_file()
def test_only_required_selected_waveform_channels_are_packaged():
 p=ROOT/'data/frozen_inputs/waveforms_selected'
 fs=list(p.glob('*.csv'));assert len(fs)==6
 prov=pd.read_csv(ROOT/'data/provenance/selected_waveforms_sha256.csv');assert len(prov)==6
 assert prov.OriginalSHA256.str.fullmatch(r'[0-9a-f]{64}').all()
 for f in fs:
  d=pd.read_csv(f,nrows=1)
  assert 'Time' in d.columns
  assert set(d.columns).issubset({'Time'}|{f'{ph}_{b}' for ph in ['Va','Ia'] for b in range(1,31)})
def test_local_reproduction_inputs_are_present():
 assert (ROOT/'data/frozen_inputs/features/ieee14_fault_generalization_features.csv.gz').is_file()
 assert (ROOT/'data/frozen_inputs/features/ieee30_fault_generalization_features.csv.gz').is_file()
 assert (ROOT/'data/frozen_inputs/predictions/resistance_holdout_predictions.csv').is_file()
 assert (ROOT/'data/frozen_inputs/predictions/heldout_10ohm_predictions.csv').is_file()
