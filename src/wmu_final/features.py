"""Frozen feature-table loading and multi-WMU spatial normalization."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / 'data/frozen_inputs/features'
EPS = 1e-12


def load_features(network: str) -> pd.DataFrame:
    """Load one packaged, frozen WMU feature table."""
    if network not in {'ieee14', 'ieee30'}:
        raise ValueError(f'unsupported network: {network}')
    path = FEATURE_DIR / f'{network}_fault_generalization_features.csv.gz'
    frame = pd.read_csv(path)
    frame['NetworkID'] = network
    for column in ['CaseID','FaultBus','WMUBus','FaultInceptionAngleDeg','NumBuses']:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(int)
    frame['FaultResistanceOhm'] = pd.to_numeric(frame['FaultResistanceOhm'], errors='coerce').astype(float)
    return frame


def add_spatial(frame: pd.DataFrame, buses: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute absolute and per-case spatially normalized V/I changes."""
    sub = frame[frame['WMUBus'].astype(int).isin([int(bus) for bus in buses])].copy()
    phase_v_pre = [c for c in ['v_pre_rms_A','v_pre_rms_B','v_pre_rms_C'] if c in sub]
    phase_v_evt = [c for c in ['v_event_rms_A','v_event_rms_B','v_event_rms_C'] if c in sub]
    phase_i_pre = [c for c in ['i_pre_rms_A','i_pre_rms_B','i_pre_rms_C'] if c in sub]
    phase_i_evt = [c for c in ['i_event_rms_A','i_event_rms_B','i_event_rms_C'] if c in sub]
    if not all((phase_v_pre, phase_v_evt, phase_i_pre, phase_i_evt)):
        raise ValueError('feature table is missing required pre/event phase RMS columns')
    sub['DeltaV_abs'] = (sub[phase_v_evt].mean(axis=1) - sub[phase_v_pre].mean(axis=1)).abs()
    sub['DeltaI_abs'] = (sub[phase_i_evt].mean(axis=1) - sub[phase_i_pre].mean(axis=1)).abs()
    audits = []
    for case_id, indices in sub.groupby('CaseID').groups.items():
        group = sub.loc[indices]
        isum, vsum = float(group['DeltaI_abs'].sum()), float(group['DeltaV_abs'].sum())
        imax, vmax = float(group['DeltaI_abs'].max()), float(group['DeltaV_abs'].max())
        sub.loc[indices, 'I_rel_sum'] = group['DeltaI_abs'] / (isum + EPS)
        sub.loc[indices, 'V_rel_sum'] = group['DeltaV_abs'] / (vsum + EPS)
        sub.loc[indices, 'I_rel_max'] = group['DeltaI_abs'] / (imax + EPS)
        sub.loc[indices, 'V_rel_max'] = group['DeltaV_abs'] / (vmax + EPS)
        sub.loc[indices, 'I_spatial_rank_norm'] = group['DeltaI_abs'].rank(method='first').sub(1) / max(len(group)-1, 1)
        sub.loc[indices, 'V_spatial_rank_norm'] = group['DeltaV_abs'].rank(method='first').sub(1) / max(len(group)-1, 1)
        sub.loc[indices, 'VI_log_delta_ratio'] = np.log10((group['DeltaV_abs'] + EPS) / (group['DeltaI_abs'] + EPS))
        sub.loc[indices, 'VI_rel_sum_ratio'] = (sub.loc[indices, 'V_rel_sum'] + EPS) / (sub.loc[indices, 'I_rel_sum'] + EPS)
        audits.append({'CaseID': int(case_id), 'WMUCount': len(group), 'DeltaISum': isum, 'DeltaVSum': vsum, 'SmallDeltaISum': isum < 1e-9, 'SmallDeltaVSum': vsum < 1e-9})
    return sub, pd.DataFrame(audits)
