import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import plotly.graph_objects as go
from datetime import datetime

from smc.indicators import SMCIndicators

def fetch_data(symbol="USTEC.r", timeframe=mt5.TIMEFRAME_M15, n_bars=1000):
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return None
        
    print(f"Buscando {n_bars} barras de {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("Nenhum dado retornado.")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def plot_smc(df: pd.DataFrame):
    print("Processando SMC...")
    smc = SMCIndicators(df)
    df_smc = smc.compute_all()
    
    print("Desenhando gráfico...")
    fig = go.Figure()
    
    # Candlestick principal
    fig.add_trace(go.Candlestick(
        x=df_smc['time'],
        open=df_smc['open'],
        high=df_smc['high'],
        low=df_smc['low'],
        close=df_smc['close'],
        name='Preço'
    ))
    
    # 1. Swings
    sh = df_smc[df_smc['is_swing_high']]
    sl = df_smc[df_smc['is_swing_low']]
    
    fig.add_trace(go.Scatter(
        x=sh['time'], y=sh['high'],
        mode='markers',
        marker=dict(color='orange', symbol='triangle-down', size=10),
        name='Swing High'
    ))
    
    fig.add_trace(go.Scatter(
        x=sl['time'], y=sl['low'],
        mode='markers',
        marker=dict(color='purple', symbol='triangle-up', size=10),
        name='Swing Low'
    ))
    
    # 2. BOS
    bos = df_smc[df_smc['is_bos']]
    for _, row in bos.iterrows():
        color = 'blue' if row['bos_direction'] == 1 else 'red'
        txt = "BOS Bull" if row['bos_direction'] == 1 else "BOS Bear"
        fig.add_trace(go.Scatter(
            x=[row['time']], y=[row['close']],
            mode='markers+text',
            marker=dict(color=color, size=12, symbol='x'),
            text=[txt],
            textposition='middle right',
            name='BOS',
            showlegend=False
        ))

    # 3. CISD
    cisd = df_smc[df_smc['is_cisd']]
    for _, row in cisd.iterrows():
        color = 'green' if row['cisd_direction'] == 1 else 'magenta'
        txt = "CISD Bull" if row['cisd_direction'] == 1 else "CISD Bear"
        fig.add_trace(go.Scatter(
            x=[row['time']], y=[row['close']],
            mode='markers+text',
            marker=dict(color=color, size=12, symbol='star'),
            text=[txt],
            textposition='middle left',
            name='CISD',
            showlegend=False
        ))
        
    # 4. FVG (Retângulos)
    # df_smc deve conter o index, precisamos de iterrows mas com acesso ao idx inteiro.
    # iterrows já dá o index da linha, então vamos usar df_smc.loc.
    fvg = df_smc[df_smc['is_fvg']]
    for idx, row in fvg.iterrows():
        if idx < 2:
            continue
            
        color = 'rgba(0, 255, 0, 0.2)' if row['fvg_direction'] == 1 else 'rgba(255, 0, 0, 0.2)'
        line_color = 'rgba(0, 255, 0, 0.5)' if row['fvg_direction'] == 1 else 'rgba(255, 0, 0, 0.5)'
        
        if row['fvg_direction'] == 1: # Bullish
            top = row['low']
            bot = df_smc.loc[idx - 2, 'high']
        else: # Bearish
            top = df_smc.loc[idx - 2, 'low']
            bot = row['high']
            
        time_start = df_smc.loc[idx - 2, 'time']
        time_end = row['time'] # Estende até o candle atual
        
        # Opcional: estender o FVG para frente na tela para fácil visualização
        time_extended = df_smc['time'].iloc[min(idx + 10, len(df_smc)-1)]
        
        fig.add_shape(
            type="rect",
            x0=time_start, y0=bot, x1=time_extended, y1=top,
            fillcolor=color, line_color=line_color, layer="below"
        )
        
    # 5. Sweeps
    sweeps = df_smc[df_smc['is_sweep']]
    for _, row in sweeps.iterrows():
        color = 'cyan' if row['sweep_direction'] == 1 else 'orange'
        y_val = row['low'] if row['sweep_direction'] == 1 else row['high']
        txt = "Sweep (Bull)" if row['sweep_direction'] == 1 else "Sweep (Bear)"
        y_anchor = 'top center' if row['sweep_direction'] == 1 else 'bottom center'
        
        fig.add_trace(go.Scatter(
            x=[row['time']], y=[y_val],
            mode='markers+text',
            marker=dict(color=color, size=12, symbol='cross'),
            text=[txt],
            textposition=y_anchor,
            name='Sweep',
            showlegend=False
        ))

    # Estética
    fig.update_layout(
        title="Auditoria Visual SMC",
        yaxis_title="Preço",
        xaxis_title="Tempo",
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800
    )
    
    fig.show()

if __name__ == "__main__":
    df = fetch_data("USTEC.r", mt5.TIMEFRAME_M15, 1000)
    if df is not None:
        plot_smc(df)
