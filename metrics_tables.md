# Сводные таблицы метрик

> Источник чисел — `metrics.txt` (прогоны пользователя). Все значения, кроме отмеченных ※, —
> **зрелые метрики: среднее по последним 2 фолдам** (`fold >= max_fold − 1`), режим `real_plus_synth`,
> метрики по дневным часам 6:00–23:00. Чем меньше — тем лучше. **Жирным** — лучший по столбцу.
> Классические baseline-модели отмечены как `real_only`: синтетика для них не используется, но фолды и дневная маска те же.
> Файл пополняется по мере появления новых прогонов в ноутбуках.

## Horizon = 1 час

| Метод | MAE | SMAPE | WAPE |
|---|---|---|---|
| Classical baseline · Clean Ensemble (`real_only`) | 38.96 | 26.90 | 15.41 |
| Classical baseline · kNN Lag (`real_only`) | 48.79 | 24.68 | 18.53 |
| Classical baseline · ETS (`real_only`) | 57.45 | 31.34 | 21.56 |
| Classical baseline · Holiday Profile (`real_only`) | 58.03 | 26.22 | 21.72 |
| Classical baseline · Same-Type Day (`real_only`) | 58.03 | 26.22 | 21.72 |
| Classical baseline · Seasonal Naive (`real_only`) | 58.03 | 26.22 | 21.72 |
| Classical baseline · Weighted Profile (`real_only`) | 268.03 | 171.43 | 91.15 |
| Classical baseline · Mean Profile (`real_only`) | 269.45 | 172.28 | 91.59 |
| TCN | 37.92 | 22.99 | 13.03 |
| N-BEATS | 33.73 | 21.42 | 12.66 |
| TFT | 36.38 | 20.08 | 13.08 |
| GAT · A — baseline EgoGAT | 33.21 | 22.95 | 12.15 |
| GAT · B — только embeddings | 32.82 | 20.88 | 11.99 |
| GAT · C — только adjacency | 29.84 | 26.71 | 13.84 |
| GAT · D — обе идеи (1+2) | 29.47 | 19.80 | 11.40 |
| GAT · D + SAN | **28.95** | 20.31 | **11.10** |
| GAT · D + TimesNet (blocks 1) | 30.49 | **18.57** | 11.84 |

## Horizon = 24 часа

| Метод | MAE | SMAPE | WAPE |
|---|---|---|---|
| Classical baseline · kNN Lag (`real_only`) | 112.13 | 50.22 | 52.11 |
| Classical baseline · Seasonal Naive (`real_only`) | 116.63 | 42.54 | 55.06 |
| Classical baseline · Holiday Profile (`real_only`) | 122.84 | 43.21 | 56.78 |
| Classical baseline · Same-Type Day (`real_only`) | 122.84 | 43.21 | 56.78 |
| Classical baseline · ETS (`real_only`) | 138.00 | 49.75 | 68.17 |
| Classical baseline · Clean Ensemble (`real_only`) | 146.10 | 51.24 | 66.57 |
| Classical baseline · Weighted Profile (`real_only`) | 184.47 | 57.63 | 84.13 |
| Classical baseline · Mean Profile (`real_only`) | 194.07 | 59.02 | 88.62 |
| TCN | 57.65 | 39.42 | 30.07 |
| N-BEATS | 61.77 | 37.36 | 30.84 |
| TFT | 46.50 | 34.08 | 26.90 |
| GAT · A — baseline EgoGAT ※ | 51.85 | 30.94 | 23.80 |
| GAT · B — только embeddings | 76.74 | 37.26 | 36.71 |
| GAT · C — только adjacency | 72.63 | 35.83 | 33.56 |
| GAT · D — обе идеи (1+2) | **39.42** | **26.08** | 20.61 |
| GAT · D + SAN | 41.41 | 27.26 | 20.68 |
| GAT · D + TimesNet (blocks 1) | 39.55 | 27.81 | **20.37** |

※ Для A на h=24 в `metrics.txt` приведён только 3-фолдовый summary (не зрелые last-2-fold метрики),
поэтому строка не вполне сопоставима с остальными и взята в скобки при сравнениях.

---

### Краткие наблюдения (по зрелым метрикам)
- **h=1:** лучший MAE/WAPE — **D + SAN (28.95 / 11.10)**, лучший SMAPE — **D + TimesNet (18.57)**; все три D-конфига опережают одиночные идеи (B, C), нейросетевые базлайны (TFT/N-BEATS/TCN) и классические baseline-модели.
- **h=24:** по MAE/SMAPE лидирует **D — обе идеи** (39.42 / 26.08), лучший WAPE — **D + TimesNet (20.37)**; против TFT (46.50 / 34.08 / 26.90) — MAE −15%, WAPE −24%. Одиночные идеи проваливаются (B 76.74, C 72.63) — только пара 1+2 регуляризует эмбеддинги графом.
- **SAN (⚖ нейтрален):** на h=1 косметический плюс по MAE/WAPE, на h=24 слегка вредит (41.41 vs 39.42). Подтверждает, что выигрыш D идёт от идей 1+2, а не от slice-нормировки.
- **TimesNet:** смещает выигрыш в **процентную/объёмную точность** — лучший SMAPE на h=1 (18.57) и лучший WAPE на h=24 (20.37), но по MAE на зрелых фолдах не превосходит чистый D (3-фолдовый «чемпионский» MAE 24.94/41.87 размывается на последних 2 фолдах). Размен «MAE↔SMAPE/WAPE».
- Общий мотив: adaptive embeddings и adjacency по отдельности нестабильны (особенно на h=24), но их **совместный конфиг D даёт синергию** и обгоняет TFT на обоих горизонтах; SAN/TimesNet добавляют точечные размены по типам метрик.
