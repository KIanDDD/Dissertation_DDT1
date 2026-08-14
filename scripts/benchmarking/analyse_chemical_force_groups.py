from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TACRINE_ATOM_COUNT = 30

REQUIRED = [
    'filename', 'atom_index_1based', 'element',
    'force_x_eV_per_A', 'force_y_eV_per_A', 'force_z_eV_per_A',
    'force_magnitude_eV_per_A',
    'mean_Fx_eV_per_A', 'mean_Fy_eV_per_A', 'mean_Fz_eV_per_A',
    'ensemble_mean_force_magnitude_eV_per_A',
    'rms_vector_disagreement_eV_per_A',
    'vector_difference_magnitude_eV_per_A',
    'force_vector_cosine_similarity',
]


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if len(df) != 7800:
        raise ValueError(f'Expected 7800 atomic rows; found {len(df)}')
    if df['filename'].nunique() != 100:
        raise ValueError('Expected 100 structures')
    counts = df.groupby('filename').size()
    if not (counts == 78).all():
        raise ValueError('Each structure must contain 78 atomic rows')
    if df['atom_index_1based'].min() != 1 or df['atom_index_1based'].max() != 78:
        raise ValueError('Atom indices must span 1-78')


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['molecular_group'] = np.where(
        out['atom_index_1based'] <= TACRINE_ATOM_COUNT,
        'Tacrine',
        'Water',
    )
    out['chemical_group'] = out['molecular_group'] + ' ' + out['element']
    expected = {'Tacrine C', 'Tacrine H', 'Tacrine N', 'Water H', 'Water O'}
    observed = set(out['chemical_group'].unique())
    if observed != expected:
        raise ValueError(
            'Unexpected atom ordering or composition. '
            f'Expected {expected}; found {observed}'
        )
    return out


def metrics(group: pd.DataFrame) -> dict[str, float | int]:
    mace = group[['force_x_eV_per_A', 'force_y_eV_per_A', 'force_z_eV_per_A']].to_numpy(float)
    aim = group[['mean_Fx_eV_per_A', 'mean_Fy_eV_per_A', 'mean_Fz_eV_per_A']].to_numpy(float)
    diff = aim - mace
    cosine = group['force_vector_cosine_similarity'].to_numpy(float)
    cosine = cosine[np.isfinite(cosine)]
    return {
        'structure_count': int(group['filename'].nunique()),
        'atomic_observation_count': int(len(group)),
        'force_component_count': int(diff.size),
        'component_mae_eV_per_A': float(np.mean(np.abs(diff))),
        'component_rmse_eV_per_A': float(np.sqrt(np.mean(diff ** 2))),
        'mean_vector_difference_eV_per_A': float(group['vector_difference_magnitude_eV_per_A'].mean()),
        'maximum_vector_difference_eV_per_A': float(group['vector_difference_magnitude_eV_per_A'].max()),
        'mean_force_vector_cosine_similarity': float(np.mean(cosine)) if cosine.size else float('nan'),
        'mean_aimnet_ensemble_force_disagreement_eV_per_A': float(group['rms_vector_disagreement_eV_per_A'].mean()),
        'maximum_aimnet_ensemble_force_disagreement_eV_per_A': float(group['rms_vector_disagreement_eV_per_A'].max()),
        'mean_mace_force_magnitude_eV_per_A': float(group['force_magnitude_eV_per_A'].mean()),
        'mean_aimnet2_force_magnitude_eV_per_A': float(group['ensemble_mean_force_magnitude_eV_per_A'].mean()),
    }


def grouped(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(column, sort=True):
        row = {column: name}
        row.update(metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def barplot(data: pd.DataFrame, category: str, value: str, title: str, output: Path) -> None:
    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111)
    ax.bar(data[category], data[value])
    ax.set_ylabel('Force-component RMSE (eV Å$^{-1}$)')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, axis='y', linewidth=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.outdir}')
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validate(df)
    df = add_groups(df)

    by_element = grouped(df, 'element')
    by_molecule = grouped(df, 'molecular_group')
    by_chemical = grouped(df, 'chemical_group')

    df.to_csv(args.outdir / 'chemical_force_records.csv', index=False)
    by_element.to_csv(args.outdir / 'force_metrics_by_element.csv', index=False)
    by_molecule.to_csv(args.outdir / 'force_metrics_tacrine_vs_water.csv', index=False)
    by_chemical.to_csv(args.outdir / 'force_metrics_by_chemical_group.csv', index=False)

    summary = {
        'status': 'complete',
        'input_file': str(args.input.resolve()),
        'structure_count': int(df['filename'].nunique()),
        'atomic_force_rows': int(len(df)),
        'assumed_atom_ordering': 'Atoms 1-30 tacrine; atoms 31-78 sixteen waters',
        'overall_metrics': metrics(df),
        'interpretation': (
            'These values quantify disagreement between MACE-OFF23 and '
            'the four-member AIMNet2 ensemble mean. They are not errors '
            'against an ab-initio reference.'
        ),
    }
    (args.outdir / 'chemical_force_analysis_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )

    barplot(by_element, 'element', 'component_rmse_eV_per_A',
            'MACE-AIMNet2 force disagreement by element',
            args.outdir / '01_force_rmse_by_element.png')
    barplot(by_chemical, 'chemical_group', 'component_rmse_eV_per_A',
            'MACE-AIMNet2 force disagreement by chemical group',
            args.outdir / '02_force_rmse_by_chemical_group.png')
    barplot(by_molecule, 'molecular_group', 'component_rmse_eV_per_A',
            'MACE-AIMNet2 disagreement: tacrine versus water',
            args.outdir / '03_force_rmse_tacrine_vs_water.png')

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111)
    ax.scatter(
        df['rms_vector_disagreement_eV_per_A'],
        df['vector_difference_magnitude_eV_per_A'],
        s=12,
        alpha=0.65,
    )
    ax.set_xlabel('AIMNet2 ensemble force disagreement (eV Å$^{-1}$)')
    ax.set_ylabel('MACE-AIMNet2 force-vector difference (eV Å$^{-1}$)')
    ax.set_title('Ensemble uncertainty versus inter-model force disagreement')
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(args.outdir / '04_uncertainty_vs_force_disagreement.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f'\nSaved outputs to: {args.outdir.resolve()}')


if __name__ == '__main__':
    main()
