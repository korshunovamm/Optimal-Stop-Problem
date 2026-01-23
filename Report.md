Задача оптимальной остановки (американский опцион) в дискретном времени: найти value function $V_0$ и стратегию стопа $\tau$.

В практической части: 
сравнить методы аппроксимации continuation value:

LSM (Longstaff–Schwartz),

RLSM / RRLSM (randomized neural nets / reservoir-style),

NLSM (нейросеть вместо базисов),

Что нового:
не только “ошибка цены”, а ошибка хеджирующей стратегии, т.е. распределение shortfall/потерь при дискретном хеджировании (маржинальные надбавки, VaR/CVaR, квантили).


Понял. Это уже **другая** проблема и она типовая для этого репозитория:

> `generate_paths() got an unexpected keyword argument 'return_dW'`

Она означает, что **часть кода (прайсер/модуль Greeks) вызывает `model.generate_paths(return_dW=...)`**, а **конкретная реализация модели `Heston` в вашем репозитории** не принимает этот аргумент (у неё другой интерфейс генерации).

Важно: это **не ваша ошибка конфигурации параметров “r, sigma, …”**, а **несовместимость флага/режима вычисления Greeks** с данной моделью.

---

## 1) Почему BS работал, а Heston падает

В репозитории разные модели реализованы неодинаково:

* **BlackScholes.generate_paths(...)** — обычно умеет возвращать и пути, и использованные приращения броуновского движения (dW), чтобы:

  * переиспользовать одинаковую случайность при finite differences,
  * корректно считать (\Delta), (\Gamma) и т.п.

* **Heston.generate_paths(...)** в вашем варианте репозитория, судя по ошибке, **не поддерживает `return_dW`**, то есть **не рассчитан на тот путь вычисления Greeks**, который включается флагами `--compute_greeks` и/или `--greeks_method`.

Именно поэтому BS проходит, а Heston падает при том же запуске.

---

## 2) Что делать, если вы НЕ хотите менять код репозитория

### Решение A (самое простое): Heston считать без Greeks в `run_algo.py`

То есть: **для Heston просто уберите `--compute_greeks`**.

```bash
python optimal_stopping/run/run_algo.py \
  --configs="diploma_minput_Heston" \
  --nb_jobs=10 \
  --print_errors
```

Это даст вам:

* цену (`price`)
* время (`comp_time`, `time_path_gen`)
  Но **без `delta`**.

Если ваша глава “pricing comparison” — это полностью покрывает задачу.

---

### Решение B (для диплома правильнее): Greeks/дельту для Heston считать отдельно, вне `--compute_greeks`

Поскольку ваша “новизна” — хеджирование, вам нужно (\Delta).
Если репозиторий не поддерживает греков для Heston “из коробки”, то корректная стратегия без правки кода:

#### Вариант B1: (\Delta) по конечным разностям на уровне “чёрного ящика”

Вы делаете **две оценки цены** тем же алгоритмом и теми же параметрами, но с разными `spot`:

[
\Delta \approx \frac{V(S_0+\varepsilon)-V(S_0-\varepsilon)}{2\varepsilon}
]

Как сделать это средствами конфигов (без правки кода):

1. В `my_configs.py` заведите **три** конфига:

   * `..._S0` (spot=100)
   * `..._Splus` (spot=100+eps)
   * `..._Sminus` (spot=100-eps)

Например (eps=0.1 или 0.01*spot, на практике 0.1–1.0 обычно устойчиво для MC):

```python
eps = 0.5

diploma_minput_Heston_S0 = _DefaultConfig(
    algos=['RLSM','NLSM','LSM'],
    stock_models=['Heston'],
    payoffs=['MinPut'],
    spots=[100],
    strikes=[100],
    maturities=[1.0],
    drift=[0.05],
    volatilities=[0.2],
    nb_dates=[50],
    nb_paths=[20000],
    nb_runs=20,
    hidden_size=[50],
    nb_epochs=[30],
    use_payoff_as_input=[True],
    train_ITM_only=[True],
)

diploma_minput_Heston_Splus = _DefaultConfig(**{**diploma_minput_Heston_S0.__dict__, "spots":[100+eps]})
diploma_minput_Heston_Sminus = _DefaultConfig(**{**diploma_minput_Heston_S0.__dict__, "spots":[100-eps]})
```

Но так напрямую с `__dict__` может быть неудобно из-за dataclass. Проще просто скопировать 2 блока и поменять `spots`.

2. Запускаете **три прогона**:

```bash
python optimal_stopping/run/run_algo.py --configs="diploma_minput_Heston_S0" --nb_jobs=10
python optimal_stopping/run/run_algo.py --configs="diploma_minput_Heston_Splus" --nb_jobs=10
python optimal_stopping/run/run_algo.py --configs="diploma_minput_Heston_Sminus" --nb_jobs=10
```

3. В ноутбуке объединяете три CSV и считаете `delta_fd` по формуле выше (по алгоритму и run-id/seed).

Да, это не так “элегантно”, как авто-греки, но **полностью воспроизводимо** и **не требует правки репозитория**, и для диплома это нормально (вы явно пишете: “дельта оценена разностным методом по двум ценам”).

---

## 3) Какая “более сложная модель” у вас реально сейчас доступна для хеджа в рамках репозитория

С учётом ограничений интерфейса:

* Для **BS**: можно использовать `--compute_greeks` и получать (\Delta) штатно (идеально для хеджа).
* Для **Heston**: либо без греков в run_algo, либо греков через внешний FD (как выше).
* Для **RoughHeston**: скорее всего ситуация аналогичная или даже сложнее (часто нет поддержки “одним флагом”).

То есть “следующая сложная модель” для диплома — **Heston**, но с пониманием, что Greeks придётся считать **внешним способом**, если вы не меняете код.

---

## 4) Что я рекомендую как траекторию для диплома (чтобы не застрять)

1. **BS / MinPut**

   * Получить цену + (\Delta) штатно (`--compute_greeks`)
   * Сделать риск хеджирования (L), VaR/CVaR и sensitivity по `nb_dates`.

2. **Heston / MinPut**

   * Сравнить цены (без greeks) между методами.
   * Для хеджа: (\Delta) через FD по цене (3 конфига spot).

3. (опционально) Rough Heston только на pricing, если успеете.

---

## Команда, которая у вас сработает прямо сейчас для Heston (без правок кода)

```bash
python optimal_stopping/run/run_algo.py \
  --configs="diploma_minput_Heston" \
  --nb_jobs=10 \
  --print_errors
```

---

Если вы хотите, я в следующем сообщении дам:

* готовый шаблон `my_configs.py` для `Heston_S0/Splus/Sminus`,
* и готовый кусок кода для ноутбука: загрузить 3 CSV, собрать средние цены по algo и посчитать `delta_fd` + доверительный интервал для delta.
