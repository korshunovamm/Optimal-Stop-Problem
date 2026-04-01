# План экспериментов: Хеджирование американских опционов на BTC с помощью нейронных сетей

## 0. Контекст и постановка задачи

### Что уже сделано (30-03-26)

На синтетических данных (Black-Scholes, Heston) показано:
- **RLSM в 86 раз быстрее LSM** при d=100 с сопоставимой или лучшей ценой
- RLSM и NLSM дают **аналитическую дельту** через autograd (не нужен finite-difference)
- При d=5 LSM пока побеждает по CVaR95 хеджа, но при росте размерности его FD-дельта деградирует

### Что нужно доказать на реальных данных

**Главный тезис диплома**: нейросетевые методы (RLSM, NLSM) обеспечивают **более качественный хедж** американских опционов на BTC, чем LSM, при сопоставимой или лучшей точности цены.

### Имеющиеся данные

| Источник | Описание | Период | Строк |
|----------|----------|--------|-------|
| `data/bybit_BTCUSDT_ohlc_interval_1min.csv` | OHLCV свечи BTCUSDT (1 мин) | 2021-07-06 — 2024-07-22 | ~1.6M |
| `data/dump_*.sql` → `options_table` | BTC/ETH опционы (strike, expiry, IV, bid/ask, underlying) | 2024-03 — 2024-06 | ~18.7M |
| `data/dump_*.sql` → `perpetuals_table` | BTC perpetual фьючерсы (funding, mark, bid/ask) | 2024-03 — 2024-11 | ~25K |

---

## 1. Как считается хедж на биткоине: детальное описание

### 1.1 Постановка

Продавец выписывает **американский пут-опцион** на BTCUSDT:
- Страйк K (например, ATM или 5-10% OTM)
- Срок T (7, 14, 30 дней)
- Дискретное исполнение: N точек ребалансировки (ежедневно или каждые 4/8 часов)

Продавец получает премию V₀ и должен хеджироваться так, чтобы покрыть обязательства при исполнении.

### 1.2 Обучение моделей

Есть два принципиально разных подхода к генерации тренировочных траекторий:

**Подход A: Calibrated Synthetic (модель-зависимый)**
1. По историческим данным BTC оцениваем параметры стохастической модели (BS или Heston)
2. Генерируем 50 000+ синтетических траекторий из калиброванной модели
3. Обучаем backward induction (LSM/RLSM/NLSM) на этих траекториях

**Подход B: Historical Bootstrap (модель-свободный)**
1. Из реальных минутных данных BTC строим ценовые пути нужной длины (resample на 4h/daily)
2. Применяем block bootstrap для создания множества псевдо-траекторий
3. Обучаем LSM/RLSM/NLSM непосредственно на исторических данных

Подход B — ключевое преимущество NN-методов: не нужно предполагать конкретный стохастический процесс.

### 1.3 Расчёт дельты

В момент времени tᵢ, наблюдая цену BTC Sᵢ, вычисляем дельту хеджа:

| Метод | Формула дельты | Вычислительная стоимость |
|-------|---------------|------------------------|
| **LSM** | Finite difference: Δ = [V(S+ε) - V(S-ε)] / 2ε | 2 полных вычисления V на каждый шаг |
| **NLSM** | `torch.autograd.grad(Q_net(S), S)` — точная производная | 1 forward + 1 backward pass |
| **RLSM** | Аналитический: Δ = W_out · ∂reservoir(S)/∂S | 1 forward pass через фиксированный reservoir |

Для RLSM и NLSM дельта — **бесплатный побочный продукт** обучения. Это ключевое преимущество.

### 1.4 Механика дискретного хеджа

```
Начальный портфель:
  cash = V₀ - Δ₀ · S₀
  stock_position = Δ₀

В каждый момент ребалансировки tᵢ (i = 1, ..., N):
  1. Наблюдаем новую цену Sᵢ
  2. Портфель: Φᵢ = cash · e^(r·dt) + stock_position · Sᵢ
  3. Стоимость предъявления: payoff(Sᵢ) = max(K - Sᵢ, 0)
  4. Shortfall: Lᵢ = e^(-r·tᵢ) · max(0, payoff(Sᵢ) - Φᵢ)
  5. Пересчитываем дельту: Δᵢ = get_delta(model, Sᵢ, tᵢ)
  6. Ребалансируем: cash = Φᵢ - Δᵢ · Sᵢ,  stock_position = Δᵢ

Итоговая метрика по траектории:
  max_shortfall = max(L₁, ..., Lₙ)  — максимальный дисконтированный убыток
```

### 1.5 Риск-метрики

По множеству тестовых траекторий (реальных или синтетических) считаем:
- **VaR₉₅**: 95-й квантиль распределения max_shortfall
- **CVaR₉₅ (λ-маржа)**: среднее значение max_shortfall в хвосте выше VaR₉₅
- **Hedge effectiveness**: 1 - Var(hedge_error) / Var(option_payoff)
- **Mean absolute shortfall**: средний абсолютный дефицит по всем траекториям
- **Sharpe ratio хеджа**: (E[PnL]) / Std[PnL] — качество хеджирующей стратегии

λ-маржа показывает, сколько дополнительного капитала нужно зарезервировать для покрытия рисков.

---

## 2. План экспериментов

### Эксперимент 1: Калибровка моделей на BTC

**Цель**: Подобрать параметры BS и Heston, отражающие реальную динамику BTC.

**Методология**:
1. Из 1-мин OHLCV данных BTC вычислить:
   - Историческую волатильность σ по скользящему окну (30d, 60d, 90d)
   - Realized variance и её автокорреляцию (для Heston)
   - Статистику доходностей: среднее, стд, скос, эксцесс (подтвердить heavy tails)
2. Калибровка Black-Scholes: σ = realized vol за окно
3. Калибровка Heston: из options_table извлечь implied volatility surface, подогнать (κ, θ, ρ, v₀) минимизацией расхождения рыночных и модельных IV
4. Альтернативная калибровка Heston: из исторической realized variance оценить mean-reversion параметры (κ, θ) и vol-of-vol (ξ)

**Файлы**: `calibrate_bs.py`, `calibrate_heston.py`

**Метрики успеха**:
- QQ-plot модельных vs реальных доходностей
- RMSE модельной vs рыночной IV surface
- p-value теста Колмогорова-Смирнова для распределения доходностей

---

### Эксперимент 2: Validation на синтетических путях с калиброванными параметрами BTC

**Цель**: Убедиться, что при параметрах BTC (высокая волатильность, stochastic vol) NN-методы по-прежнему работают.

**Параметры**:
- Put1Dim: K = S₀ = 65000, σ = 0.7 (калибр.), r = 0.05, T = 30d
- N = 30 (ежедневная ребалансировка)
- 50000 тренировочных путей, 5000 тестовых

**Алгоритмы**: LSM, RLSM, NLSM

**Зачем**: Проверить, что при реалистичных BTC-параметрах (σ≈0.5-1.0, heavy tails у Heston) алгоритмы корректно считают цену и хедж. Это "мост" между синтетикой из 30-03-26 и реальными данными.

**Файлы**: `run_exp2_btc_synthetic_validation.py`

**Метрики**: Цена vs binomial reference, CVaR₉₅ хеджа, fit time

---

### Эксперимент 3: Historical Bootstrap — обучение на реальных BTC путях

**Цель**: Обучить LSM/RLSM/NLSM непосредственно на исторических ценах BTC, без предположения о модели генерации.

**Методология**:
1. **Подготовка данных**:
   - Из 1-мин свечей агрегировать до 4h/daily close prices
   - Нормализация: разделить на S₀ окна (работаем с относительными ценами S/S₀)
   - Определить «опцион»: American Put, K = S₀ (ATM), T = 30 дней, N = 30 шагов (daily)
2. **Создание тренировочных путей** (block bootstrap):
   - Из 3 лет дневных данных (~1100 дней) вырезать блоки по 30 дней
   - Со сдвигом в 1 день → ~1070 перекрывающихся путей
   - Для аугментации: добавить paths из калиброванной модели (Experiment 2)
3. **Backward induction на реальных путях**:
   - Заменяем `model.generate_paths()` на массив реальных путей
   - `fit_backward_snapshots` работает с любым массивом (nb_paths, nb_stocks, nb_dates+1)
4. **Дельта-хедж на отложенных данных**:
   - Последние 6 месяцев данных (~2024-01 — 2024-07) — test set
   - Для каждого 30-дневного окна: обучаем на предыдущих данных, хеджируем на тестовом окне

**Файлы**: `btc_data_loader.py`, `run_exp3_historical_bootstrap.py`

**Метрики**: CVaR₉₅ хеджа, hedge effectiveness, сравнение с Exp 2

---

### Эксперимент 4: Rolling Backtest — полный хеджирующий бэктест на BTC

**Цель**: Имитировать реальную торговую стратегию хеджирования на исторических данных.

**Методология** (rolling window):
```
Для каждой даты t в [2023-01-01, 2024-07-01]:
  1. Определяем опцион: American Put, K = BTC_close(t), T = 30d
  2. Тренировочное окно: BTC prices за [t - 365d, t]
  3. Обучаем LSM/RLSM/NLSM на historical bootstrap из окна
  4. Запускаем дельта-хедж на реальных ценах BTC [t, t+30d]
  5. Фиксируем: max shortfall, terminal PnL, все Δ(tᵢ)
```

**Ребалансировка**: ежедневная (N=30 для T=30d), также тест с N=7 (еженедельная)

**Ключевые режимы рынка в тестовом периоде**:
- Боковик (~30-40k, 2023H1)
- Бычий рынок (40k→73k, 2023Q4-2024Q1)
- Коррекция (73k→57k, 2024Q2)
- Восстановление (57k→67k, 2024Q2-Q3)

Это позволяет оценить робастность хеджа в разных рыночных фазах.

**Файлы**: `run_exp4_rolling_backtest.py`

**Метрики**:
- Cumulative hedge PnL curve для каждого алгоритма
- CVaR₉₅ по всем 30-дневным окнам
- Breakdown по рыночным режимам (bull/bear/sideways)
- Hedge effectiveness = 1 - Var(shortfall) / Var(payoff)

---

### Эксперимент 5: Сравнение скорости дельта-вычислений (inference time)

**Цель**: Количественно показать, что NN-дельта быстрее FD-дельты при реальном использовании.

**Методология**:
- Зафиксировать обученные модели из Exp 3/4
- Для 10 000 случайных точек (S, t) измерить время:
  - LSM FD delta (2 evaluations per point)
  - RLSM analytical delta (1 forward pass)
  - NLSM autograd delta (1 forward + backward pass)
- Повторить для d=1 и для обогащённого признакового пространства (d>1, см. Exp 7)

**Файлы**: `run_exp5_delta_speed_benchmark.py`

**Метрики**: Среднее время на 1 delta (мкс), speedup ratio, scaling с d

---

### Эксперимент 6: Ablation — NLSM vs RLSM vs различные архитектуры

**Цель**: Определить, какой NN-метод лучше всего подходит для BTC.

**Варианты**:
1. **RLSM (baseline NN)**: фиксированный reservoir, линейный выход
2. **NLSM (полная NN)**: обучаемые все слои
3. **RLSM с разным hidden_size**: 64, 128, 256, 512
4. **RLSM с разными activation**: LeakyReLU, ReLU, Tanh
5. **NLSM с разным числом эпох**: 50, 100, 200, 500

Все на одних и тех же BTC-данных из Exp 3.

**Файлы**: `run_exp6_ablation.py`

**Метрики**: Цена, CVaR₉₅ хеджа, fit time, delta inference time

---

### Эксперимент 7: Многомерное состояние (фичи-обогащение) — d > 1

**Цель**: Показать преимущество NN в ситуации, когда пространство признаков расширяется.

**Идея**: Для 1D BTC-опциона расширяем входное пространство дополнительными признаками:
1. d=1: только цена BTC
2. d=2: цена BTC + realized volatility (за 30 дней)
3. d=3: цена BTC + realized vol + volume
4. d=5: + funding rate (из perpetuals) + open interest (из options)
5. d=8: + BTC returns за 1d, 7d, 14d

При d=5+ LSM сталкивается с проклятием размерности (полиномиальный базис взрывается), а RLSM работает стабильно.

**Важно**: Это главный аргумент за NN. На реальном рынке одной цены недостаточно — нужно учитывать множество факторов. LSM не может этого сделать при d>3-4, а NN — может.

**Адаптация кода**:
- Создать `HistoricalMultiDimModel` — загружает BTC цену + доп. фичи как nb_stocks > 1
- Payoff остаётся Put1Dim по первой компоненте (цена BTC)
- backward_snapshots уже поддерживает nb_stocks > 1

**Файлы**: `btc_feature_builder.py`, `run_exp7_multidim_features.py`

**Метрики**: Цена, CVaR₉₅ по каждому d, speedup RLSM vs LSM

---

### Эксперимент 8: Использование реальной IV surface для валидации

**Цель**: Сравнить модельную цену с рыночной ценой BTC-опциона.

**Методология**:
1. Из `options_table` выбрать BTC пут-опционы с T≈30d, moneyness ≈ ATM
2. Зафиксировать `mark_price` и `mid_IV` как рыночный benchmark
3. На дату observation: обучить LSM/RLSM/NLSM на данных до этой даты
4. Сравнить модельную цену V₀ с `mark_price`
5. Implied volatility из модельной цены vs рыночная `mid_IV`

**Файлы**: `run_exp8_market_iv_validation.py`

**Метрики**: Ошибка цены (abs, relative), ошибка IV

---

## 3. Предполагаемые результаты

### 3.1 Ожидания по экспериментам

| Эксперимент | Ожидаемый результат | Почему |
|-------------|-------------------|--------|
| Exp 1 (калибровка) | σ_BTC ≈ 0.5-0.8, тяжёлые хвосты, vol clustering | Характерно для крипто |
| Exp 2 (BS/Heston BTC-params) | RLSM ≈ LSM по цене, NLSM чуть хуже | При d=1 LSM хорош, но дельта NN точнее |
| Exp 3 (historical bootstrap) | NN адаптируется лучше к non-stationarity | Нет model misspecification |
| **Exp 4 (rolling backtest)** | **RLSM: CVaR₉₅ на 10-30% ниже LSM** | Более точная дельта, быстрая адаптация |
| Exp 5 (speed) | RLSM delta в 5-50x быстрее LSM FD | 1 pass vs 2 evaluations |
| Exp 6 (ablation) | RLSM hidden=256 оптимален | Баланс expressiveness и overfitting |
| **Exp 7 (multidim features)** | **При d≥5 LSM деградирует, RLSM стабилен** | Проклятие размерности LSM |
| Exp 8 (market IV) | Модельная цена в пределах bid-ask spread | Валидация адекватности |

### 3.2 Почему NN должны быть лучше для хеджа на BTC

1. **Скорость дельты**: RLSM даёт delta за 1 forward pass. При реальной торговле время — деньги (рынок уходит, slippage растёт).

2. **Точность дельты**: Autograd даёт **точную** производную, а не numerical approximation. При высокой волатильности BTC (σ≈0.7) finite difference с фиксированным bump ε нестабилен.

3. **Многомерность**: Реальный хедж должен учитывать не только цену, но и volatility regime, funding rate, order flow. LSM не масштабируется на d>3 из-за O(d²) базисных функций.

4. **Нестационарность**: BTC — нестационарный процесс с режимами (bull/bear/sideways). NN лучше аппроксимирует сложные нелинейные continuation values, чем полиномы LSM.

5. **Online переоценка**: Обученный RLSM не нужно переобучать для вычисления delta. Reservoir фиксирован, линейный слой пересчитывается за O(1). Это критично для practice: модель обучена раз в день, а delta считается каждую секунду.

### 3.3 Риски и ограничения

| Риск | Как митигировать |
|------|-----------------|
| Model misspecification (BS/Heston не описывают BTC jumps) | Exp 3 (bootstrap) обходит это |
| Overfitting NN на малом числе путей | Cross-validation, regularization |
| Transaction costs при частой ребалансировке | Учесть спред и slippage в PnL |
| Non-stationarity — сдвиг распределения | Rolling window recalibration |
| Низкая ликвидность BTC опционов | Использовать synthetic options |

---

## 4. Архитектура кода

### 4.1 Структура директории `31-03-26-bybit/`

```
31-03-26-bybit/
├── plan.md                         # этот файл
├── config_btc.py                   # параметры экспериментов для BTC
├── btc_data_loader.py              # загрузка и подготовка BTC OHLCV
├── btc_feature_builder.py          # создание multi-dim фичей (vol, volume, funding)
├── btc_options_loader.py           # загрузка опционных данных из SQL dump
├── calibrate_bs.py                 # калибровка BS на BTC (Exp 1)
├── calibrate_heston.py             # калибровка Heston на BTC IV surface (Exp 1)
├── historical_model.py             # HistoricalModel — адаптер для реальных путей
├── run_exp2_btc_synthetic.py       # Exp 2: синтетика с BTC-параметрами
├── run_exp3_historical_bootstrap.py # Exp 3: обучение на исторических BTC
├── run_exp4_rolling_backtest.py    # Exp 4: rolling window бэктест
├── run_exp5_delta_speed.py         # Exp 5: benchmark скорости дельты
├── run_exp6_ablation.py            # Exp 6: ablation study
├── run_exp7_multidim_features.py   # Exp 7: многомерное пространство фичей
├── run_exp8_market_validation.py   # Exp 8: валидация на рыночных IV
├── generate_figures.py             # визуализация результатов
└── output/                         # CSV, JSON, PNG результаты
```

### 4.2 Ключевая адаптация: HistoricalModel

Для работы с `fit_backward_snapshots` необходим объект, совместимый с интерфейсом `Model`. Создаём адаптер:

```python
class HistoricalModel:
    """Wraps real BTC price paths to be compatible with the library's Model interface."""
    def __init__(self, paths, rate, maturity, nb_dates):
        # paths: np.ndarray shape (nb_paths, nb_stocks, nb_dates+1)
        self.nb_paths = paths.shape[0]
        self.nb_stocks = paths.shape[1]
        self.nb_dates = nb_dates
        self.maturity = maturity
        self.rate = rate
        self.spot = float(paths[0, 0, 0])
        self.dividend = 0.0
        self.volatility = 0.0  # не используется напрямую
        self.name = "Historical"
        self._paths = paths

    def generate_paths(self):
        return self._paths.copy(), None
```

Это позволяет передать реальные BTC-пути в `fit_backward_snapshots` без изменения кода backward induction.

### 4.3 Подготовка данных BTC

```python
def load_btc_daily(csv_path, start_date=None, end_date=None):
    """Load 1-min BTC OHLCV, resample to daily close."""
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    if start_date: df = df[df.index >= start_date]
    if end_date: df = df[df.index <= end_date]
    daily = df['price_close'].resample('1D').last().dropna()
    return daily

def create_paths_sliding_window(prices, window_size=30, step=1):
    """Create overlapping paths from a price series.
    
    Returns: np.ndarray shape (nb_paths, 1, window_size+1)
    Prices are normalized: S(t)/S(0) * S0_reference
    """
    paths = []
    for i in range(0, len(prices) - window_size, step):
        segment = prices[i : i + window_size + 1]
        # Нормализация: в единицах начальной цены
        normalized = segment / segment[0] * REFERENCE_SPOT
        paths.append(normalized)
    paths = np.array(paths)  # (nb_paths, window_size+1)
    return paths[:, np.newaxis, :]  # (nb_paths, 1, window_size+1)
```

---

## 5. Визуализации (generate_figures.py)

### 5.1 Обязательные графики

1. **BTC price + realized vol time series** — контекст данных
2. **QQ-plot модельных vs реальных доходностей** (Exp 1)
3. **Calibrated model paths overlay на реальные** (Exp 1)
4. **Цена опциона: LSM vs RLSM vs NLSM на BTC-синтетике** (Exp 2)
5. **Распределение hedge errors для каждого алгоритма** (Exp 3, 4) — гистограммы
6. **Cumulative hedge PnL по времени** (Exp 4) — equity curve
7. **VaR/CVaR comparison bar chart** (Exp 3, 4)
8. **Delta inference time bar chart** (Exp 5)
9. **CVaR₉₅ vs dimension d** для LSM vs RLSM (Exp 7) — показывает деградацию LSM
10. **Hedge effectiveness breakdown по рыночным режимам** (Exp 4)

### 5.2 Таблицы для диплома

| Таблица | Содержание |
|---------|-----------|
| T1 | Калиброванные параметры BS/Heston для BTC по периодам |
| T2 | Цены опциона: LSM vs RLSM vs NLSM vs market (Exp 2, 8) |
| T3 | Риск-метрики хеджа: VaR₉₅, CVaR₉₅, mean shortfall, hedge effectiveness |
| T4 | Время обучения и inference для каждого алгоритма |
| T5 | Ablation: влияние hidden_size и nb_epochs на качество (Exp 6) |
| T6 | CVaR₉₅ хеджа при разных d (Exp 7) |

---

## 6. Порядок выполнения

### Phase 1: Подготовка данных и калибровка (Exp 1)
1. `btc_data_loader.py` — загрузка и предобработка BTC OHLCV
2. `calibrate_bs.py` — оценка σ скользящим окном
3. `calibrate_heston.py` — подгонка Heston к realized variance
4. Визуализация: QQ-plots, vol time series

### Phase 2: Синтетическая валидация (Exp 2)
5. `run_exp2_btc_synthetic.py` — цена и хедж при BTC-параметрах
6. Сравнительная таблица с результатами из 30-03-26

### Phase 3: Historical training и backtest (Exp 3, 4)
7. `historical_model.py` — адаптер HistoricalModel
8. `run_exp3_historical_bootstrap.py` — обучение на реальных путях
9. `run_exp4_rolling_backtest.py` — rolling window хедж-бэктест
10. Визуализация: equity curves, hedge error distributions

### Phase 4: Анализ скорости и ablation (Exp 5, 6)
11. `run_exp5_delta_speed.py` — benchmark
12. `run_exp6_ablation.py` — подбор гиперпараметров

### Phase 5: Многомерность и рыночная валидация (Exp 7, 8)
13. `btc_feature_builder.py` — построение multi-dim фичей
14. `run_exp7_multidim_features.py` — показать масштабируемость NN
15. `btc_options_loader.py`, `run_exp8_market_validation.py` — сравнение с рынком

### Phase 6: Визуализация и отчёт
16. `generate_figures.py` — все графики
17. Обновление `plan.md` с результатами

---

## 7. Критерии успеха

| Критерий | Порог | Комментарий |
|----------|-------|-------------|
| CVaR₉₅(RLSM) < CVaR₉₅(LSM) | На 10%+ | Ключевой результат для диплома |
| Delta speed: RLSM < LSM | В 5x+ | Практическая применимость |
| Hedge effectiveness(RLSM) > 0.7 | > 70% | Хедж покрывает большую часть риска |
| При d≥5: LSM деградирует, RLSM нет | CVaR разрыв > 30% | Масштабируемость |
| Модельная цена vs market mark_price | Ошибка < 5% | Адекватность модели |

---

## 8. Связь с теорией (статья Optimal-Stopping-Via-Randomized-NN)

### Что берём из статьи
- **Theorem 1 (Convergence)**: RLSM сходится к истинной цене при достаточном hidden_size. Это обосновывает применение на реальных данных: если мы увеличиваем reservoir, аппроксимация улучшается.
- **Section 4 (Numerical experiments)**: статья показывает RLSM на MaxCall d=2-200. Мы расширяем на BTC — актив с нестандартным распределением.
- **Key insight**: RLSM обучается без gradient descent (только lstsq), что делает его устойчивым к выбору learning rate и числа эпох. На нестационарных BTC-данных это преимущество.

### Что добавляем (новизна диплома)
- **Хеджирующая стратегия**: статья НЕ рассматривает хедж. Мы показываем, что RLSM дельта (через reservoir autograd) работает для практического хеджирования.
- **Реальные данные**: статья работает только с синтетикой. Мы тестируем на BTC.
- **Распределение ошибки хеджа**: анализ VaR/CVaR как λ-маржи — прямая практическая ценность.
- **Feature-enriched state space**: d>1 для BTC с дополнительными рыночными фичами — LSM проигрывает.
