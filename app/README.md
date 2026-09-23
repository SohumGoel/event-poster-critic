---
title: Event Poster Critic
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Event Poster Critic

Upload a draft event poster and get predicted human ratings (1–10) for **alignment, overlap and white space**, plus tips for the weakest areas.

- **Data:** [GraphicDesignEvaluation](https://huggingface.co/datasets/creative-graphic-design/GraphicDesignEvaluation), with ratings from 60 human judges.
- **Model:** a frozen image model plus regressors we trained to predict the average human rating (see `poster_critic.ipynb`).
