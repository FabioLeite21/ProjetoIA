import pandas as pd
import numpy as np

import pandas as pd

def filter_fixation_events(df):
    # Selecionar apenas colunas com tipo 'object' (strings)
    string_cols = df.select_dtypes(include=['object']).columns
    df_filtered = df.copy()
    
    # Filtrar linhas onde nenhuma das colunas de string contém os termos
    mask = ~df_filtered[string_cols].apply(lambda row: row.str.contains('Total|Average|Count|Frequency', na=False).any(), axis=1)
    df_filtered = df_filtered[mask]
    
    return df_filtered

def infer_timestamps(df_events):
    """
    Infere timestamps cumulativos a partir das durations, assumindo sequência de eventos.
    Começa do tempo 0.
    """
    current_time = 0.0  # Em ms
    df_events['start_time'] = np.nan
    df_events['end_time'] = np.nan
    
    for idx, row in df_events.iterrows():
        fixation_duration = row['Fixation Duration [ms]']
        df_events.at[idx, 'start_time'] = current_time
        df_events.at[idx, 'end_time'] = current_time + fixation_duration
        current_time += fixation_duration  # Adicionar saccade duration se separado
    
    # Se houver saccade durations separadas, ajuste aqui para incluir entre fixações
    return df_events