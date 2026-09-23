"""Event Poster Critic: upload a draft event poster, get feedback calibrated to human ratings.

Learned relationship: frozen image embedding -> our regressors -> predicted average human rating (1-10)
for alignment, overlap and white space. The regressors are trained in poster_critic.ipynb on
GraphicDesignEvaluation (ratings from 60 human judges) and saved to model.joblib.
"""
import gradio as gr
import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor

final = joblib.load("model.joblib")
backbone = AutoModel.from_pretrained(final["backbone_id"]).eval()
processor = AutoImageProcessor.from_pretrained(final["backbone_id"])

THRESHOLD = 6.0  # scores below this get a tip
TIPS = {
    "alignment": "Elements don't line up. Snap text and images to a shared grid or left edge.",
    "overlap": "Elements collide. Keep text off busy image areas and separate overlapping blocks.",
    "whitespace": "Spacing feels off. Add margins and breathing room, or cut content so it isn't crammed.",
}
LABELS = {"alignment": "Alignment", "overlap": "Overlap", "whitespace": "White space"}


@torch.no_grad()
def embed(img):
    f = backbone.get_image_features(**processor(images=[img.convert("RGB")], return_tensors="pt"))
    f = f if torch.is_tensor(f) else f.pooler_output  # transformers >= 5 returns an output object
    return torch.nn.functional.normalize(f.float(), dim=-1).numpy()


def critique(img):
    if img is None:
        return {}, "Upload a poster to get feedback."
    x = embed(img)
    scores = {p: float(np.clip(m.predict(x)[0], 1, 10)) for p, m in final["models"].items()}
    weak = sorted((s, p) for p, s in scores.items() if s < THRESHOLD)
    if weak:
        lines = [f"**{LABELS[p]}: {s:.1f}/10.** {TIPS[p]}" for s, p in weak]
        feedback = "### What to fix first\n" + "\n".join(f"{i}. {l}" for i, l in enumerate(lines, 1))
    else:
        feedback = "### Looks solid\nAlignment, overlap and white space all score 6 or above."
    table = "\n".join(f"| {LABELS[p]} | {s:.1f} / 10 |" for p, s in scores.items())
    feedback += f"\n\n| Principle | Predicted human rating |\n|---|---|\n{table}"
    return {LABELS[p]: s / 10 for p, s in scores.items()}, feedback


ABOUT = f"""
**Who it's for:** students and club organizers designing event posters, at the draft-critique stage.

**How it works:** a frozen image model (`{final["backbone"]}`) turns your poster into a vector. Our regressors,
trained on the [GraphicDesignEvaluation](https://huggingface.co/datasets/creative-graphic-design/GraphicDesignEvaluation)
dataset, predict the **average rating 60 human judges** would give it (1–10) for each design principle.
Scores below {THRESHOLD:g} come with a fix.

**Limitations:** it was trained on web graphics and banners rather than printed posters, it covers only 3 principles
(not color, typography or content), and human raters themselves disagree, so treat the scores as guidance.
"""

with gr.Blocks(title="Event Poster Critic") as demo:
    gr.Markdown("# Event Poster Critic\nUpload a draft event poster to see how human judges would likely rate "
                "its **alignment, overlap and white space**, and what to fix first.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Your event poster", height=480)
            btn = gr.Button("Critique my poster", variant="primary")
        with gr.Column():
            scores_out = gr.Label(label="Predicted human rating (out of 10)", num_top_classes=3)
            feedback_out = gr.Markdown()
    with gr.Accordion("About this tool", open=False):
        gr.Markdown(ABOUT)
    btn.click(critique, inp, [scores_out, feedback_out])
    inp.upload(critique, inp, [scores_out, feedback_out])

if __name__ == "__main__":
    demo.launch()
