# -*- coding: utf-8 -*-
"""
MetaTrader 5 Client for TCC Quant Trading.

Módulo de abstração de dados de altíssima fidelidade.
Responsável por estabelecer comunicação com a instância local do terminal
MetaTrader 5, extraindo histórico de OHLCV (nativas do corretor) para
timeframes diversos (ex: M15) preservando spread e volume em ticks.
"""

import logging
import pandas as pd
from datetime import datetime

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("Por favor, instale o MetaTrader5 usando 'pip install MetaTrader5'.")

logger = logging.getLogger(__name__)

class MT5Client:
    """
    Cliente para integração e extração de dados do MetaTrader 5.
    
    Gerencia de forma segura o encerramento do backend nativo em C++ no `__del__`.
    """
    def __init__(self):
        # Tenta inicializar. Se o terminal não estive aberto localmente, possivelmente errará.
        if not mt5.initialize():
            error_code = mt5.last_error()
            logger.error(f"Falha ao inicializar o MetaTrader 5. Código de erro: {error_code}")
            mt5.shutdown()
            raise ConnectionError("MetaTrader5 não pôde ser iniciado. O software está aberto e logado?")
        
        # Validar versão da lib para confirmar
        v = mt5.version()
        logger.info(f"Conexão com MetaTrader 5 estabelecida com sucesso. Terminal version: {v}")
        
    def __del__(self):
        """Assegura o fechamento sustentável do conector para não vazar mem."""
        mt5.shutdown()
        logger.info("MetaTrader 5 connection closed.")
        
    def get_historical_data(self, symbol: str, timeframe: int, n_bars: int) -> pd.DataFrame:
        """
        Extrai as ultimas 'n_bars' do timeframe selecionado usando C++ Bridge do MT5.
        
        Args:
            symbol (str): O ticker nativo do ativo na corretora em vigor (ex: 'EURUSD', 'WINM24').
            timeframe (int): Constante nativa do MT5 (ex: mt5.TIMEFRAME_M15).
            n_bars (int): Quantidade de barras (puxa da ponta mais atual até n_bars atrás).
        
        Returns:
            pd.DataFrame: Dataframe no formato nativo pro smc/indicators.py (lowercase).
        """
        logger.info(f"Baixando {n_bars} barras do ativo '{symbol}'...")
        
        # Verifica se o símbolo existe na lista do Market Watch da corretora conectada
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Ativo {symbol} não foi encontrado. Verifique se a string está correta e visível no Market Watch.")
            return pd.DataFrame()
            
        if not symbol_info.visible:
            logger.info(f"Ativando {symbol} no Market Watch...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Falhou ao selecionar o {symbol}.")
                return pd.DataFrame()

        # copy_rates_from_pos extrai do index 0 (mais recente) retroativamente
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Nenhum dado baixado. mt5.last_error(): {mt5.last_error()}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        
        # O MT5 retorna timestamp unix inteiro na coluna 'time'
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # O DataFrame original retornado pelo mt5 traz colunas como:
        # 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'
        expected_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']
        valid_cols = [c for c in expected_cols if c in df.columns]
        df = df[valid_cols]
        
        logger.info(f"Dados históricos adquiridos. Linhas: {len(df)}")
        return df
