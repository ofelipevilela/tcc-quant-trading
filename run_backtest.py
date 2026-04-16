# -*- coding: utf-8 -*-
"""
Script Orquestrador de Backtesting (SMC + ANFIS).

Baixa dados de um ativo real (ex: S&P 500 via yfinance), roda o pipeline de
indicadores SMC, injeta os dados no modelo ANFIS treinado, e processa um mock
de carteira gerando a curva de equidade final.
"""

import logging
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import MetaTrader5 as mt5
    from data.mt5_client import MT5Client
except ImportError:
    print("Por favor, instale o MetaTrader5.")
    sys.exit(1)

from backtest.engine import BacktestEngine
from backtest.performance import calculate_performance_metrics
from smc.feature_factory import FEATURE_MODE_LEGACY_LIKE, resolve_feature_mode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_backtest")


def load_model_metadata(model_path: str) -> dict:
    """Carrega metadados gerados no treino, se existirem."""
    meta_path = Path(model_path).with_name(f"{Path(model_path).stem}_meta.json")
    if not meta_path.exists():
        logger.warning("Metadados do modelo nao encontrados em '%s'. Usando defaults.", meta_path)
        return {}

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info("Metadados carregados de '%s'.", meta_path)
        return metadata
    except Exception as exc:
        logger.warning("Falha ao ler metadados do modelo: %s", exc)
        return {}

def main():
    # 1. Configuração e download de dados (MT5)
    ticker_name = "USTEC.r"  # Mude pro ativo que você operar no MT5
    timeframe = mt5.TIMEFRAME_M15
    n_bars = 5000
    
    logger.info(f"Baixando dados do {ticker_name} (M15, {n_bars} barras)...")
    mt5_client = MT5Client()
    df = mt5_client.get_historical_data(ticker_name, timeframe, n_bars)
    
    if len(df) == 0:
        logger.error("Nenhum dado baixado. Verifique a conexao MT5 ou o ticker.")
        return

    # Normalizar as colunas do yfinance para lowercase e remover multi-index se existir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower().strip() for c in df.columns]
    else:
        df.columns = [c.lower().strip() for c in df.columns]


    # 2. Caminho do modelo treinado do MT5
    model_path = "anfis_trained_mt5.pt"
    metadata = load_model_metadata(model_path)
    activation_threshold = float(metadata.get("recommended_threshold", 0.20))
    max_holding_bars = int(metadata.get("horizon", 15))
    atr_mult = float(metadata.get("atr_mult", 1.0))
    swing_window = int(metadata.get("swing_window", 5))
    feature_mode = resolve_feature_mode(
        metadata.get("feature_mode"),
        default=FEATURE_MODE_LEGACY_LIKE,
    )

    logger.info(
        "Config backtest carregada: threshold=%.2f | horizon=%d | atr_mult=%.2f | swing_window=%d | feature_mode=%s",
        activation_threshold,
        max_holding_bars,
        atr_mult,
        swing_window,
        feature_mode,
    )
    
    # 3. Inicializa e roda o motor
    logger.info("Inicializando o motor de backtesting...")
    # O threshold absoluto agora segue o intervalo real da rede em [-1, 1].
    engine = BacktestEngine(
        data=df,
        model_path=model_path,
        initial_capital=100000.0,
        risk_per_trade=0.01,
        reward_to_risk=1.0,
        activation_threshold=activation_threshold,
        stop_mode="atr",
        atr_stop_mult=atr_mult,
        atr_target_mult=atr_mult,
        max_holding_bars=max_holding_bars,
        smc_swing_window=swing_window,
        smc_feature_mode=feature_mode,
    )
    
    equity_curve, trades = engine.run()
    
    # 4. Avaliar métricas
    metrics = calculate_performance_metrics(equity_curve, trades, timestamps=df.index)
    
    logger.info("\n" + "="*40)
    logger.info("=== RESULTADOS DO BACKTEST ===")
    logger.info("="*40)
    logger.info(f"Retorno Total:   {metrics['total_return']:.2f}%")
    logger.info(f"Ret. Anualiz.:   {metrics['annualized_return']:.2f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    logger.info(f"Win Rate:        {metrics['win_rate']:.2f}% ({metrics['winning_trades']} wins / {metrics['losing_trades']} losses)")
    logger.info(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    logger.info(f"Payoff Ratio:    {metrics['payoff_ratio']:.2f}")
    logger.info(f"Expectancy:      {metrics['expectancy']:.2f}")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio:   {metrics['sortino_ratio']:.2f}")
    logger.info(f"Calmar Ratio:    {metrics['calmar_ratio']:.2f}")
    logger.info(f"SQN:             {metrics['sqn']:.2f}")
    logger.info(f"Total de Trades: {metrics['total_trades']}")
    if trades:
        exit_counts = pd.Series([trade.get('exit_reason', 'UNKNOWN') for trade in trades]).value_counts().to_dict()
        avg_bars = float(pd.Series([trade.get('bars_held', 0) for trade in trades]).mean())
        logger.info(f"Long / Short:    {metrics['long_trades']} / {metrics['short_trades']}")
        logger.info(f"Ret. Medio:      {metrics['avg_trade_return_pct']:.3f}%")
        logger.info(f"Desv. Trade:     {metrics['trade_return_std_pct']:.3f}%")
        logger.info(f"Exit Reasons:    {exit_counts}")
        logger.info(f"Tempo Medio:     {avg_bars:.2f} barras")
    logger.info("="*40)
    
    # 5. Visualizar
    if len(equity_curve) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Preço e Entradas
        ax1.plot(df.index, df['close'], label='Close Price', color='white', alpha=0.6)
        
        # Opcional: Adicionar marcações dos trades
        # Para cada trade, colocar um scatter
        df_trades_long = [t for t in trades if t['type'] == 'LONG']
        df_trades_short = [t for t in trades if t['type'] == 'SHORT']
        
        if df_trades_long:
            idx_l = [t['entry_date'] for t in df_trades_long]
            price_l = [t['entry_price'] for t in df_trades_long]
            ax1.scatter(idx_l, price_l, color='#00FF41', marker='^', s=100, label='Long Entry', zorder=5)
            
        if df_trades_short:
            idx_s = [t['entry_date'] for t in df_trades_short]
            price_s = [t['entry_price'] for t in df_trades_short]
            ax1.scatter(idx_s, price_s, color='#FF003C', marker='v', s=100, label='Short Entry', zorder=5)

        ax1.set_title(f"Price Action e Decisões do ANFIS ({ticker_name})", color='white')
        ax1.set_facecolor('#0f0f13')
        ax1.grid(color='#2a2a35', linestyle=':', alpha=0.5)
        ax1.legend()
        
        # Plot 2: Curva de Equidade
        ax2.plot(df.index, equity_curve, color='#00FF41', linewidth=2, label='Equity Curve ($)')
        ax2.fill_between(df.index, engine.initial_capital, equity_curve, 
                         where=(pd.Series(equity_curve) >= engine.initial_capital), 
                         color='#00FF41', alpha=0.1)
        ax2.fill_between(df.index, engine.initial_capital, equity_curve, 
                         where=(pd.Series(equity_curve) < engine.initial_capital), 
                         color='#FF003C', alpha=0.1)
                         
        ax2.axhline(engine.initial_capital, color='gray', linestyle='--', alpha=0.7)
        ax2.set_title(f"Evolução do Capital (Ini: ${engine.initial_capital:.2f})", color='white')
        ax2.set_facecolor('#0f0f13')
        ax2.grid(color='#2a2a35', linestyle=':', alpha=0.5)
        ax2.set_ylabel("Capital USD", color='white')
        ax2.legend()
        
        fig.patch.set_facecolor('#0f0f13')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='gray')
            for spine in ax.spines.values():
                spine.set_color('#2a2a35')

        plt.tight_layout()
        plt.savefig("outputs/plots/backtest_results.png", dpi=300, bbox_inches='tight')
        logger.info("Gráfico do backtest salvo em 'outputs/plots/backtest_results.png'")
        # plt.show() # bloqueia a execução

if __name__ == "__main__":
    main()
