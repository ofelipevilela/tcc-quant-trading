# -*- coding: utf-8 -*-
"""
Teste de validação do SMC Indicators com dados reais.

Baixa dados do S&P 500 via yfinance, roda o pipeline SMC completo
e verifica que os outputs estão dentro dos universos esperados pelo ANFIS.
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('smc_test_log.txt', encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
L = logging.getLogger('smc_test')

try:
    import yfinance as yf
    import numpy as np

    from smc.indicators import SMCIndicators

    # ====================================================================
    # 1. Baixar dados
    # ====================================================================
    L.info("Baixando dados do S&P 500 (1 ano, diario)...")
    ticker = yf.Ticker("^GSPC")
    df = ticker.history(period="1y", interval="1d")
    L.info(f"OK: {len(df)} barras baixadas")
    L.info(f"Colunas: {list(df.columns)}")
    L.info(f"Periodo: {df.index[0]} a {df.index[-1]}")

    # ====================================================================
    # 2. Rodar pipeline SMC
    # ====================================================================
    L.info("Executando pipeline SMC...")
    smc = SMCIndicators(df, swing_window=5)
    result = smc.compute_all()
    L.info(f"Pipeline completo: {len(result)} barras processadas")

    # ====================================================================
    # 3. Resumo
    # ====================================================================
    stats = smc.summary()
    L.info("=== RESUMO SMC ===")
    for k, v in stats.items():
        if isinstance(v, float):
            L.info(f"  {k:25s} = {v:.4f}")
        else:
            L.info(f"  {k:25s} = {v}")

    # ====================================================================
    # 4. Validar universos ANFIS
    # ====================================================================
    anfis_df = smc.get_anfis_inputs()
    L.info("=== VALIDAÇÃO ANFIS INPUTS ===")

    checks = {
        'trend_strength': (-100.0, 100.0),
        'price_zone':     (0.0, 1.0),
        'fvg_quality':    (0.0, 4.0),
        'sweep_quality':  (0.0, 3.0),
    }

    all_ok = True
    for col, (lo, hi) in checks.items():
        col_min = anfis_df[col].min()
        col_max = anfis_df[col].max()
        col_mean = anfis_df[col].mean()
        n_nan = anfis_df[col].isna().sum()
        within = (col_min >= lo) and (col_max <= hi) and (n_nan == 0)
        status = "OK" if within else "FALHA"
        if not within:
            all_ok = False
        L.info(
            f"  {col:20s}: min={col_min:8.4f}, max={col_max:8.4f}, "
            f"mean={col_mean:8.4f}, NaN={n_nan}, [{status}]"
        )

    L.info("")
    if all_ok:
        L.info("TODOS OS INPUTS DENTRO DOS UNIVERSOS ANFIS!")
    else:
        L.info("ATENCAO: Algum input fora do universo esperado.")

    # ====================================================================
    # 5. Amostra de dados
    # ====================================================================
    L.info("\n=== AMOSTRA (últimas 10 barras) ===")
    sample = result[['close', 'trend_strength', 'price_zone', 'fvg_quality',
                      'sweep_quality', 'is_bos', 'bos_direction',
                      'is_fvg', 'is_sweep']].tail(10)
    L.info(f"\n{sample.to_string()}")

    L.info("\nTESTE CONCLUIDO COM SUCESSO!")

except Exception as e:
    L.error(f"ERRO: {e}")
    import traceback
    L.error(traceback.format_exc())
