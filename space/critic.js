// Event Poster Critic: scoring logic that runs in the browser.
// Frozen image model (transformers.js) -> unit-length embedding -> our trained regressors (weights.json).

const ONNX_IDS = {
  "google/siglip-base-patch16-224": "Xenova/siglip-base-patch16-224",
  "openai/clip-vit-base-patch32": "Xenova/clip-vit-base-patch32",
  "openai/clip-vit-large-patch14": "Xenova/clip-vit-large-patch14",
};

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function predict(m, x) {
  if (m.type === "linear") return dot(m.coef, x) + m.intercept;
  // StandardScaler + MLP (ReLU hidden layers, linear output), matching sklearn's MLPRegressor
  let h = x.map((v, i) => (v - m.mean[i]) / m.scale[i]);
  m.layers.forEach((layer, li) => {
    const out = layer.b.slice();
    for (let i = 0; i < h.length; i++) {
      const row = layer.W[i], hi = h[i];
      for (let j = 0; j < out.length; j++) out[j] += hi * row[j];
    }
    h = li < m.layers.length - 1 ? out.map(v => Math.max(0, v)) : out;
  });
  return h[0];
}

// Bicubic resize that reproduces Pillow / torchvision (antialiased, a = -0.5, 22-bit fixed point).
// transformers.js resizes slightly differently, which shifts embeddings enough to change scores by ~1 point.
function cubic(x) {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function coeffs(inSize, outSize) {
  const PREC = 22, scale = inSize / outSize, fs = Math.max(1, scale), support = 2 * fs;
  const bounds = [], kernels = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = [];
    let ww = 0;
    for (let x = 0; x < xmax; x++) { const w = cubic((x + xmin - center + 0.5) / fs); k.push(w); ww += w; }
    kernels.push(k.map(w => { const v = (w / ww) * (1 << PREC); return v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5); }));
    bounds.push(xmin);
  }
  return { bounds, kernels };
}

function resamplePass(src, w, h, c, outW, outH, horizontal) {
  const { bounds, kernels } = coeffs(horizontal ? w : h, horizontal ? outW : outH);
  const out = new Uint8ClampedArray(outW * outH * c), half = 1 << 21, div = 1 << 22;
  for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) {
    const i = horizontal ? x : y, k = kernels[i], start = bounds[i];
    for (let ch = 0; ch < c; ch++) {
      let s = half;
      for (let t = 0; t < k.length; t++) {
        const px = horizontal ? (y * w + start + t) : ((start + t) * w + x);
        s += k[t] * src[px * c + ch];
      }
      out[(y * outW + x) * c + ch] = Math.floor(s / div);
    }
  }
  return out;
}

function resizeBicubic(rgb, w, h, outW, outH) {
  let data = rgb;
  if (outW !== w) { data = resamplePass(data, w, h, 3, outW, h, true); w = outW; }
  if (outH !== h) { data = resamplePass(data, w, h, 3, outW, outH, false); }
  return data;
}

function toPixelValues(T, image, size, mean, std) {
  const img = image.rgb();
  const px = resizeBicubic(img.data, img.width, img.height, size, size);
  const out = new Float32Array(3 * size * size), n = size * size;
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) {
    out[ch * n + i] = (px[i * 3 + ch] / 255 - mean[ch]) / std[ch];
  }
  return new T.Tensor("float32", out, [1, 3, size, size]);
}

export async function loadCritic(T, weights, { dtype = "fp32", progress } = {}) {
  const id = ONNX_IDS[weights.backbone_id];
  if (!id) throw new Error("No browser version of " + weights.backbone_id);
  const isSiglip = id.includes("siglip");
  const Model = isSiglip ? T.SiglipVisionModel : T.CLIPVisionModelWithProjection;
  const model = await Model.from_pretrained(id, { dtype, progress_callback: progress });
  // SigLIP: squash to 224x224, mean/std 0.5. CLIP: shortest side to 224 then center crop, CLIP mean/std.
  const prep = isSiglip
    ? img => toPixelValues(T, img, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    : img => {
        const s = 224 / Math.min(img.width, img.height);
        const w = Math.round(img.width * s), h = Math.round(img.height * s);
        const rgb = img.rgb(), px = resizeBicubic(rgb.data, rgb.width, rgb.height, w, h);
        const x0 = Math.floor((w - 224) / 2), y0 = Math.floor((h - 224) / 2);
        const crop = new Uint8ClampedArray(224 * 224 * 3);
        for (let y = 0; y < 224; y++) crop.set(px.subarray(((y + y0) * w + x0) * 3, ((y + y0) * w + x0 + 224) * 3), y * 224 * 3);
        return toPixelValues(T, new T.RawImage(crop, 224, 224, 3), 224,
          [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]);
      };

  return async function score(image) {
    const out = await model({ pixel_values: prep(image) });
    const x = Array.from((isSiglip ? out.pooler_output : out.image_embeds).data);
    const norm = Math.sqrt(dot(x, x));
    const xn = x.map(v => v / norm);
    const scores = {};
    for (const [p, m] of Object.entries(weights.models)) {
      scores[p] = Math.min(10, Math.max(1, predict(m, xn)));
    }
    return scores;
  };
}
