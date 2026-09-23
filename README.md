# Event Poster Critic

Upload a draft event poster and get predicted human ratings (1–10) for **alignment, overlap and white space**, plus a fix tip for each weak area.

**Live demo:** https://huggingface.co/spaces/ssg1/event-poster-critic

## How it works
```
poster → SigLIP (frozen) → 768 features → 3 trained regressors → 3 scores (1–10)
```
- **Data:** [GraphicDesignEvaluation](https://huggingface.co/datasets/creative-graphic-design/GraphicDesignEvaluation), 1,200 designs rated by human judges. The target is the average rating.
- **Model:** one regressor per principle (MLP for alignment, ridge for overlap and white space), selected with 5-fold cross-validation grouped by design.
- **Deployment:** a free HF Static Space. SigLIP runs in the browser via transformers.js, and the regressors are loaded from `weights.json`.

## Run it yourself
1. Open `poster_critic.ipynb` in Colab and pick **Runtime → Change runtime type → T4 GPU**.
2. In the first settings cell, optionally change `HF_SPACE_NAME` (default `event-poster-critic`).
3. **Runtime → Run all.** Steps 1–8 need no account or token.
4. **Step 9 (deploy) needs a Hugging Face Write token.** Create one at huggingface.co → Settings → Access Tokens → **Write**. Then either save it as a Colab secret named `HF_TOKEN` (🔑 in the sidebar) or paste it when prompted.
   - Step 9 deploys to `https://huggingface.co/spaces/<your-username>/<HF_SPACE_NAME>`, which is **always under your own account**, so you can't overwrite a teammate's Space.
   - If `space/` isn't next to the notebook (e.g. you uploaded only the notebook to Colab), upload `space/index.html`, `space/critic.js` and `space/README.md` to your Space by hand.
5. Never commit your token or paste it into a cell.

## Trained model files
Both files hold the **same trained model** (SigLIP-B/16 features; MLP for alignment, ridge for overlap and white space):
- `app/model.joblib`: the Python version saved by the notebook (step 5), used by the Gradio app. It needs scikit-learn 1.9.1 to load.
- `space/weights.json`: the same model exported to JSON (step 9) so the browser can run it. This is what the live Static Space uses.

## Repo layout
| Path | What it is |
|---|---|
| `poster_critic.ipynb` | Training notebook (Colab, T4 GPU): data, model comparison, final model, deployment |
| `space/` | The live Static Space (`index.html`, `critic.js`, `weights.json`) |
| `app/` | Gradio version of the app (needs HF PRO to host). `app/model.joblib` is the trained model from the notebook (scikit-learn 1.9.1) |
| `project.txt` | Assignment brief |
| `qr_code.png` | QR code linking to the live demo |
