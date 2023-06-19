from typing import Optional
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import ticker
import matplotlib.pyplot as plt
import sys
import os


def convert_to_latex_table(
        df: pd.DataFrame,
        row_names: list[str], 
        sort_by: list[str], 
        label: str, 
        save_path: str,
        caption: str = 'TO-DO',
        index_product_lists: list[list[str]] = [['Paraliż', 'Połączone', 'Własne'], ['$\mu$', '$\sigma$']],
        index_names: list[str] = ['Cechy', 'Wartość'],
        precision: int = 4,
        highlight_max: bool = True
    ):
    df = df.sort_values(by=sort_by)
    index = pd.MultiIndex.from_product(index_product_lists, names=index_names)
    data = df[['f1_mean', 'f1_std']].to_numpy()
    data = data.reshape((len(row_names), -1))
    stats_df = pd.DataFrame(data, index=row_names, columns=index)
    s = stats_df.style.format(precision=precision)
    if highlight_max:
        s = s.highlight_max(axis=0,
                            props='textbf:--rwrap;')
    s.to_latex(os.path.join(save_path, f'{label}.tex'),
               position='h!',
               position_float="centering",
               hrules=True,
               multicol_align='c',
               label=f'tab:{label}',
               caption=caption)


sns.set_context('talk')
sns.set_style('whitegrid')


def add_baseline(ax: plt.Axes, df: Optional[pd.DataFrame], estimator: str = 'Stratified', label: str = 'Losowa predykcja'):
    if df is not None:
        value = df[df['estimator'] == estimator].iloc[0]['f1_mean']
        ax.axhline(value, label=label, linestyle='--', color='red')
        ax.legend()


def modify_legend(ax: plt.Axes, title: str, legend_mapper: Optional[dict[str, str]]):
    handles, labels = ax.get_legend_handles_labels()
    if legend_mapper:
        labels = [legend_mapper[label]
                  if label in legend_mapper else label for label in labels]
    ax.legend(title=title, handles=handles, labels=labels, loc='lower left')


def control_savefig(name: str, path: Optional[str] = None):
    if path:
        plt.savefig(os.path.join(path, name), bbox_inches='tight')


def barplot_classifiers(df: pd.DataFrame, filtered_col: str = 'aggregation', filtered_value: str = 'avg', hue: str = 'ref_points', dummy_df: Optional[pd.DataFrame] = None, dummy_estimator: str = 'Stratified', dummy_label: str = 'Losowa predykcja', legend_mapper: Optional[dict[str, str]] = None, save_path: Optional[str] = None):
    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    df = df[df[filtered_col] == filtered_value]
    ax = sns.barplot(data=df,
                     x='estimator', y='f1', ax=ax, hue=hue, ci='sd')
    ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(visible=True, which='minor', color='grey', linewidth=0.2)
    ax.set_xlabel('Estimator')
    ax.set_ylabel('F1 measure')
    # ax.set_title(
    #     f'Jakość klasyfikacji w zależności od wybranego zbioru cech i modelu')
    # ax.tick_params(axis='x', rotation=-30)
    add_baseline(ax, dummy_df, dummy_estimator, dummy_label)
    modify_legend(ax, '', legend_mapper)
    control_savefig(f'features_estimators_{filtered_col}_{filtered_value}.png', save_path)

