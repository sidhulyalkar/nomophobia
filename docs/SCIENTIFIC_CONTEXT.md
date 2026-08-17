# Scientific context: nomophobia, smartphone addiction, and this Kaggle task

## The name

**NOMOPHOBIA** expands to **NO MObile PHone PhoBIA**. This repository takes its name from the paper:

> Sudip Bhattacharya, Md Abu Bashar, Abhay Srivastava, Amarjeet Singh. *NOMOPHOBIA: NO MObile PHone PhoBIA.* PMCID: PMC6510111; PMID: 31143710.

https://pmc.ncbi.nlm.nih.gov/articles/PMC6510111/

The paper describes nomophobia as fear or anxiety associated with being detached from mobile-phone connectivity. It also emphasizes that the presentation can overlap with anxiety, panic, social phobia, and other psychological conditions.

## Important distinction

This Kaggle competition asks us to predict a **synthetic binary smartphone-addiction label** from aggregate behavioral features. That target is not a validated clinical nomophobia diagnosis.

Accordingly:

- the paper motivates the project name and the broader behavioral question;
- it does **not** define the Kaggle target;
- competition features should not be interpreted as clinical diagnostic criteria;
- model performance should not be translated into statements about individual mental health.

## Why the distinction is useful scientifically

One of the most interesting lessons from this competition is that predictive information and causal/clinical meaning can diverge sharply. Synthetic data can contain frequency, discretization, and missingness structure that boosts leaderboard AUC without corresponding to a plausible psychological mechanism.

That is why this project tracks two questions separately:

1. **Prediction:** what produces the best honest OOF ranking?
2. **Interpretation:** what behavioral relationships are plausible versus synthetic-data artifacts?

Keeping those questions separate prevents a strong competition model from masquerading as a clinical theory.
