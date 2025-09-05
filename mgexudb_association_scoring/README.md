# MGEx Association Scoring (Uterus gene expression)

This repository computes Association Scores (AS) that quantify how strongly a gene is associated with a condition (disease/treatment) vs normal in uterus epithelial tissue. 

## What this does

Given four reliability score files (3 columns each):
- `normal_transcribed.tsv`   → ST_N
- `normal_dormant.tsv`       → SD_N
- `<cond>_transcribed.tsv`   → ST_C
- `<cond>_dormant.tsv`       → SD_C

we compute:
- **Direction A (N→C off):** `AS = ST_N + SD_C`
- **Direction B (C→N off):** `AS = ST_C + SD_N`

We then normalize (min–max to [1,100]), optionally merge A+B with **IQR-banded scaling**, and plot.


## Quickshare
1) Compute Direction A (N->C off):
python src/association_score.py \
  --normal-transcribed data/normal_transcribed.tsv \
  --normal-dormant    data/normal_dormant.tsv \
  --cond-transcribed  data/disease_transcribed.tsv \
  --cond-dormant      data/disease_dormant.tsv \
  --direction A \
  --out results/comb_1a_as.tsv \
  --plot results/comb_1a.png \
  --title "AS Scaled 1 vs Normal transcribed & Disease dormant"

2) Compute Direction B (C->N off):
