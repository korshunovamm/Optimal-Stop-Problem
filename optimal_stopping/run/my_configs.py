import configs
from configs import *
from configs import _DefaultConfig
from dataclasses import dataclass
import numpy as np



# ======================
# Pricing Stage — Scenario A: Base case
# ======================
diploma_pricing_A_base_BS_put1d = _DefaultConfig(
    stock_models=['BlackScholes'],
    payoffs=['MinPut'],     # 1D put via MinPut with nb_stocks=1
    nb_stocks=[1],
    spots=[100], strikes=[100], maturities=[1.0],
    volatilities=[0.2],
    dividends=[0.0],
    drift=[0.05],           # r-q
    nb_dates=[50],
    nb_paths=[20000],
    nb_runs=20,
    algos=['LSM', 'RLSM', 'NLSM'],
    # оставляем как в репо-стиле:
    use_payoff_as_input=[True],     # можно True; если хотите — зафиксируйте везде одинаково
    train_ITM_only=[True],
    use_path=[False],
    hidden_size=[20],       # фиксируем, чтобы NLSM/RLSM сравнивались корректно
    nb_epochs=[30],
    representations=['TablePriceDuration'],
)

# ======================
# Pricing Stage — Scenario B: Vary number of paths (M)
# ======================
diploma_pricing_B_pathsweep_BS_put1d = _DefaultConfig(
    stock_models=['BlackScholes'],
    payoffs=['MinPut'],
    nb_stocks=[1],
    spots=[100], strikes=[100], maturities=[1.0],
    volatilities=[0.2],
    dividends=[0.0],
    drift=[0.05],
    nb_dates=[50],                     # фиксируем n
    nb_paths=[2000, 5000, 20000, 100000],  # меняем M
    nb_runs=20,
    algos=['LSM', 'RLSM', 'NLSM'],
    use_payoff_as_input=[True],
    train_ITM_only=[True],
    use_path=[False],
    hidden_size=[20],
    nb_epochs=[30],
    representations=['TablePriceDuration'],
)


# ======================
# Pricing Stage — Scenario C: Vary number of dates (n)
# ======================
diploma_pricing_C_datesweep_BS_put1d = _DefaultConfig(
    stock_models=['BlackScholes'],
    payoffs=['MinPut'],
    nb_stocks=[1],
    spots=[100], strikes=[100], maturities=[1.0],
    volatilities=[0.2],
    dividends=[0.0],
    drift=[0.05],
    nb_dates=[25, 50, 100, 200],     # меняем n
    nb_paths=[20000],                # фиксируем M
    nb_runs=20,
    algos=['LSM', 'RLSM', 'NLSM'],
    use_payoff_as_input=[True],
    train_ITM_only=[True],
    use_path=[False],
    hidden_size=[20],
    nb_epochs=[30],
    representations=['TablePriceDuration'],
)



















# ======================
# Diploma: 1D American Put (Put1Dim)
# ======================

# (используйте вашу _DefaultConfig или _VerySmallDimensionTable — лучше _DefaultConfig)
# Я делаю на базе _DefaultConfig, чтобы было прозрачно.

diploma_minput1d_BS = _DefaultConfig(
    stock_models=['BlackScholes'],
    payoffs=['MinPut'],
    nb_stocks=[1],
    spots=[100],
    strikes=[100],
    maturities=[1.0],
    nb_dates=[50],
    volatilities=[0.2],
    dividends=[0.0],
    drift=[0.05],
    nb_paths=[20000],
    nb_runs=20,
    algos=['LSM','RLSM','NLSM'],
    use_payoff_as_input=[True, False],   # можно как в репо
    train_ITM_only=[True],
    representations=['TablePriceDuration'],
)

diploma_put1d_BS = _DefaultConfig(
    stock_models=['BlackScholes'],
    payoffs=['Put1Dim'],      # 1D American put
    nb_stocks=[1],
    spots=[100],              # S0
    strikes=[100],            # K
    maturities=[1.0],         # T (в годах)
    nb_dates=[50],            # n
    volatilities=[0.2],       # sigma
    dividends=[0.0],          # q
    drift=[0.05],             # r-q (если q=0 => r)
    nb_paths=[20000],         # M
    nb_runs=20,               # для CI
    algos=['LSM', 'RLSM', 'NLSM'],
    hidden_size=[50],         # для (R)LSM/NLSM (если используется)
    nb_epochs=[30],           # для NLSM (если используется)
    train_ITM_only=[True],
    use_payoff_as_input=[False],   # <-- поменять
    representations=['TablePriceDuration'],
)

diploma_put1d_Heston = _DefaultConfig(
    stock_models=['Heston'],
    payoffs=['Put1Dim'],
    nb_stocks=[1],
    spots=[100],
    strikes=[100],
    maturities=[1.0],
    nb_dates=[50],
    volatilities=[0.2],     # часто используется как начальная/базовая вола
    dividends=[0.0],
    drift=[0.05],
    # параметры Heston (по вашей схеме): mean, speed, correlation, factors
    mean=[0.04],            # theta (долгосрочная дисперсия/вола^2 — зависит от реализации)
    speed=[2.0],            # kappa
    correlation=[-0.3],     # rho
    factors=[(1.0,1.0,1.0)],# в репо это часто “vol-of-vol” и т.п. (зависит от реализации)
    nb_paths=[20000],
    nb_runs=20,
    algos=['LSM', 'RLSM', 'NLSM'],
    hidden_size=[50],
    nb_epochs=[30],
    train_ITM_only=[True],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

diploma_put1d_RoughHeston = _DefaultConfig(
    stock_models=['RoughHeston'],
    payoffs=['Put1Dim'],
    nb_stocks=[1],
    spots=[100],
    strikes=[100],
    maturities=[1.0],
    nb_dates=[50],
    volatilities=[0.2],
    dividends=[0.0],
    drift=[0.05],
    hurst=[0.1],            # H (у вас rough примеры ставят 0.05)
    factors=[(0.0008, 0.11)], # как в ваших rough-конфигах
    nb_paths=[20000],
    nb_runs=20,
    algos=['RLSM', 'NLSM'], # LSM может быть неуместен/медленен, но можно добавить
    hidden_size=[50],
    nb_epochs=[30],
    train_ITM_only=[False], # rough часто лучше без ITM-only (как у вас в примере)
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)
