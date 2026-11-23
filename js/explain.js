// Small "?" buttons that open a plain-language explainer with the math typeset.
// Any element with class "xbtn" and data-x="<key>" opens EXPLAIN[key].

export const EXPLAIN = {
  T: {
    title: 'Temperature T',
    body: `<p>Each completion gets a quality score
      $$\\kappa(y) = p(y)^{1/T},$$
      where $p(y)$ is the model's probability of the whole completion.</p>
      <ul>
        <li><b>Small $T$</b> exaggerates differences in likelihood, so the most likely completions win (quality).</li>
        <li><b>Large $T$</b> flattens them: every completion counts about equally, and only how different they are matters (diversity).</li>
      </ul>
      <p>Drag it and watch the arrows in the plot grow and shrink.</p>`,
  },
  subset: {
    title: 'How the subset is chosen',
    body: `<p>Think of each completion as an arrow. Its <b>direction</b> is its meaning (the embedding $\\phi(y)$) and its <b>length</b> is its quality $\\sqrt{\\kappa(y)}$. A set $S$ scores the squared volume its arrows span:
      $$\\det \\mathbf L_S = \\Big(\\prod_{y\\in S}\\kappa(y)\\Big)\\,\\det\\!\\big(\\Phi_S^\\top \\Phi_S\\big).$$
      The volume is large when the arrows are long (likely completions) and point different ways (different meanings), and zero when two are the same. This is a determinantal point process (DPP).</p>
      <p><b>Most likely</b> picks the set with the largest volume, adding one completion at a time. Exact search is NP-hard; the greedy answer is within a factor $(K!)^2$ of the best.</p>
      <p><b>Sample</b> draws a random set with probability proportional to its volume, so you see different good sets. Press Resample to draw again.</p>
      <p>Both modes use the same weights here, so the picture stays the same when you switch and only the selected set changes. The paper's Sample uses importance weights $p^{1/T}/q$ at $T' = 1$. They are available under Paper settings in expert mode.</p>`,
  },
  volume: {
    title: 'Spanned volume',
    body: `<p>The grey box is spanned by the selected arrows. Its squared volume, $\\det \\mathbf L_S$, is the subset's score: long arrows (likely completions) pointing in different directions (different meanings) make a big box; near-duplicates flatten it.</p>
      <p>The plot has 3 axes and the embeddings have hundreds of dimensions, so the box drawn is an approximation. The axes are fitted so that the sets the DPP tends to pick keep their volumes as closely as three dimensions allow. The readout prints the drawn volume next to the true one. Set Axes to Fit to selection for an exact box.</p>
      <p>The box needs $K = 2$ or $3$ selected completions: with the default $K = 3$ the plot shows a box. A larger $K$ (the K control, or adding completions by clicking the margins of the similarity matrix) gives a surface instead. In the 2D view a box with $K = 3$ is drawn flat.</p>
      <p>With more than 3 selected completions, a K-dimensional volume cannot be drawn in 3D. The plot stretches an open surface, like a cloth, over the tips of the arrows. The readout prints its area and the true volume.</p>`,
  },
  logp: {
    title: 'log P(S)',
    body: `<p>How probable this subset is among all size-$K$ subsets of the completions:
      $$P(S) = \\frac{\\det \\mathbf L_S}{\\sum_{|S'| = K} \\det \\mathbf L_{S'}}.$$
      The most likely subset is the one that maximizes it.</p>`,
  },
  quality: {
    title: 'Quality',
    body: `<p>The average $\\log p(y)$ over the subset: how likely the model finds these completions. Closer to 0 is more likely.</p>`,
  },
  diversity: {
    title: 'Diversity',
    body: `<p>How different the selected completions are: the log of the squared volume their unit-length embeddings span,
      $$\\log\\det\\big(\\Phi_S^\\top\\Phi_S\\big).$$</p>
      <p>It is at most 0, reached when the completions point in completely different directions, and goes to $-\\infty$ as two of them become duplicates.</p>`,
  },
  ess: {
    title: 'Effective sample size',
    body: `<p>Every DPP weight is a target divided by $q$, the probability of drawing that completion for the pool: $w = \\text{target}/q$. The effective sample size is the Kish formula $(\\sum_i w_i)^2 / \\sum_i w_i^2$ over all $N$ draws of the pool (duplicate completions included), for the weight in use. It says how many draws effectively count. It equals $N$ when all weights are equal, and falls when a few completions dominate.</p>
      <p>The targets are $p^{1/T}$ for Importance, $q\\,p^{1/T}$ for Quality, and $q$ for Unit (so its ESS is $N$).</p>`,
  },
  Tprime: {
    title: 'Importance target T (sampling temperature T′)',
    body: `<p>Controls how the completions are generated: every next-token distribution is raised to the power $1/T'$ and renormalized. $T' = 1$ samples the model as it is. Higher $T'$ gives more varied but less coherent text, which is why diversity comes from choosing the subset afterwards instead.</p>
      <p>For the examples, the 1.0 / 1.5 switch picks which precomputed pool is shown.</p>`,
  },
  scale: {
    title: 'Vector length',
    body: `<p>This sets how long each arrow is drawn. It changes the picture only, never the selection. The lock below DPP weight ties it to the weight the DPP uses, so the drawn box is the one the DPP scores. Unlock it to draw something else.</p>
      <p><b>Quality</b> draws length $\\sqrt{\\kappa(y)}$ with $\\kappa(y) = p(y)^{1/T}$, relative to the best completion in the pool. <b>Importance</b> draws $\\sqrt{\\kappa(y)/q(y)}$. <b>Unit</b> draws every completion at unit length, which shows diversity alone.</p>`,
  },
  weight: {
    title: 'DPP weight',
    body: `<p>The weight $w(y)$ sets how much each completion counts in the DPP: the kernel is $L_{ij} = \\sqrt{w_i}\\,\\phi_i\\cdot\\phi_j\\,\\sqrt{w_j}$, so a set scores the squared volume of the arrows $\\sqrt{w(y)}\\,\\phi(y)$.</p>
      <ul>
        <li><b>Importance</b>: the target is $p^{1/T}$, so $w = p^{1/T}/q$. The paper uses it for Sample.</li>
        <li><b>Quality</b>: the target is $q\\,p^{1/T}$, so $w = p^{1/T}\\,q/q = p^{1/T}$. The paper uses it for Most likely.</li>
        <li><b>Unit</b>: the target is $q$, so $w = q/q = 1$. This is pure diversity and $T$ has no effect.</li>
      </ul>
      <p>Sample divides by $q$ because the pool was drawn from the model at temperature $T'$, so frequent completions would otherwise count twice. Switching the picking mode moves the weight to the paper's weight for the new mode, unless you picked a different one. With Importance in Most likely mode, the greedy search maximizes the importance-weighted volume. With Quality in Sample mode, the pool's own sampling frequency is not corrected.</p>`,
  },
  matrices: {
    title: 'Primal and dual matrices',
    body: `<p>Stack all $N$ rows $\\sqrt{w_i}\\,\\phi_i$ of the pool into an $N \\times D$ matrix $B$. The <b>primal</b> matrix $L = B B^\\top$ is $N \\times N$ and holds the kernel between every pair of completions. The <b>dual</b> matrix $C = B^\\top B$ is $D \\times D$, with $D$ the embedding size.</p>
      <p>Both have the same nonzero eigenvalues. Sampling through the dual costs in $D$ instead of $N$.</p>
      <p>The selected set $S$ is a $K \\times K$ submatrix $L_S$ of $L$ (its rows and columns), and $\\det L_S$ is the squared volume it spans, so $\\sqrt{\\det L_S}$ is the volume.</p>
      <p><b>Primal order.</b> Blocks puts completions with similar embeddings next to each other (average-linkage clustering on $1 - \\cos$, leaves ordered so neighbors are as similar as possible). Cluster uses the clusters of the View section, then weight. The Blocks order depends only on the embeddings, so changing $T$, the weight, or the selection changes colors and ticks but never the order.</p>
      <p><b>Dual basis.</b> Principal axes rotates the embedding coordinates to the eigenvectors $V$ of the unweighted $\\Phi^\\top\\Phi$, largest variance first, and shows $C' = V_r^\\top C V_r$ for the first $r$ axes. With the Unit weight $C'$ is diagonal. Other weights add off-diagonal entries: the weighting rotates the kernel away from the pool's own axes. Embedding coordinates shows $C$ itself.</p>`,
  },
  expert: {
    title: 'Expert mode',
    body: `<p>Off, the page shows the picking mode, $T$, $K$ and Resample, two models, and the two matrices. On, it adds smaller models, the paper settings, the DPP weight and vector length, the axes and cone options, clustering choices, and the extra matrix views. The sections open by default; a section you close stays closed. Your choice is remembered in this browser.</p>`,
  },
  model: {
    title: 'Language model and embedder',
    body: `<p>The <b>model</b> writes the completions. The <b>embedder</b> turns each completion into a vector, and the angles between vectors decide which completions count as similar. They are independent choices: the same completions can be compared under different embedders.</p>`,
  },
  K: {
    title: 'Subset size K',
    body: `<p>How many completions the subset has. A larger $K$ shows more of the pool's variety but each completion adds less new meaning. $K$ cannot exceed the number of distinct completions.</p>`,
  },
  resample: {
    title: 'Resample',
    body: `<p>Draws another random subset of the same size with the same weights. It is active in Sample mode. Most likely always returns the same set for the same $T$.</p>`,
  },
  sizeq: {
    title: 'Point size shows quality',
    body: `<p>Draws every vector at unit length and lets the size of each ball show quality instead: the radius is proportional to $\\sqrt{\\kappa(y)}$ with $\\kappa = p^{1/T}$, relative to the likeliest completion, with a floor so every point stays visible. The drawn volume is then the volume of the unit vectors.</p>`,
  },
  primal: {
    title: 'Pairs of completions',
    body: `<p>The matrix $L = B B^\\top$ has one entry per pair of completions: how similar they are, scaled by their weights. Bright blocks are groups of similar completions. Completions are ordered so that similar ones sit next to each other, and the tree in the margins shows how they merge. Colored ticks mark the selected set. Click a margin position to add or remove a completion.</p>`,
  },
  dual: {
    title: 'Pairs of embedding directions',
    body: `<p>The matrix $C = B^\\top B$ has one entry per pair of embedding directions, here expressed in the pool's principal axes (the directions of largest variation, axis 1 first). Under the Unit weight it is diagonal. Other weights add off-diagonal entries. It shows no block structure. It has the same nonzero eigenvalues as the matrix of pairs of completions.</p>`,
  },
  pca: {
    title: 'Axes',
    body: `<p>Embeddings have hundreds of dimensions (384 for MiniLM-L6) and the plot shows 3. The axes are fixed for a pool: they do not change with the selection, the picking mode, $T$ or $K$, so a different set is a different choice in the same space.</p>
      <p>The 3D positions are fitted so that the sets the DPP tends to pick (the most likely sets over the range of $T$, and samples at a few values of $T$) keep their volumes as closely as three dimensions allow. For the examples the fit is rotated to a shared frame, so the view stays put when you switch models. The bars show the share of this pool's spread on each axis. For your own prompts the axes are ordered by spread. For the examples the axes are shared across models, so they are not always in decreasing order. Click 1, 2 or 3 to change the number of axes.</p>`,
  },
  span: {
    title: 'Axes',
    body: `<p><b>Fixed</b> keeps the same axes for the whole pool, whatever the selection, picking mode, $T$ or $K$.</p>
      <p><b>Fit to selection</b> chooses the axes to contain the selected arrows (an orthonormal basis of their span, for $K \\le 3$), completed with the main directions of the other completions. The drawn box is then exactly the one the DPP scores. The axes move with the selection, so points glide when you change $T$, $K$ or the picking mode.</p>`,
  },
  embedder: {
    title: 'Embedder',
    body: `<p>The embedding model decides which completions count as similar, so it sets the similarity kernel $\\Phi^\\top\\Phi$. This demo starts with Qwen3-Embedding. The paper uses MiniLM-L6 for Ambrosia and FineWeb and Qwen3-Embedding for GSM8K. "LM's own states" uses the language model's hidden states.</p>
      <p>The completions and their probabilities do not change. Only the similarity between them does, so the selected set, the clusters and the axes can change.</p>`,
  },
  clustering: {
    title: 'Clustering',
    body: `<p>Clusters only color the completions. They never affect the DPP.</p>
      <p><b>Far apart</b> (default) merges completions step by step so that no cluster contains two completions farther apart than its diameter, which gives distant points different colors. <b>Average</b> balances that with cluster size. <b>k-means</b> builds compact groups around centers, so an isolated completion can be absorbed by the nearest group.</p>
      <p>Every completion gets a color. With Far apart or Average, a completion far from all others becomes its own small cluster. Space chooses whether distances come from the embeddings or from the 3D positions you see.</p>`,
  },
  ghost: {
    title: 'Unit positions',
    body: `<p>The faint dots mark each completion's direction at unit length, where it would sit with no quality scaling. The solid points are the same completions scaled by the chosen vector length.</p>`,
  },
  preset: {
    title: 'Paper settings',
    body: `<p>These buttons set the picking mode, $T'$, $K$ and $T$ to operating points from the paper (Ambrosia, $K = 3$, 200 completions).</p>
      <ul>
        <li><b>MAP, paper</b>: Most likely on pools drawn at $T' = 1.5$. The paper sweeps $T \\in \\{1, 3, 7.5, 10\\}$; the button sets $T = 3$.</li>
        <li><b>Samp, paper</b>: Sample on pools drawn at $T' = 1.0$. The paper sweeps $T \\in \\{0.43, 0.54, 0.67, 0.82, 1\\}$; the button sets $T = 0.82$.</li>
      </ul>
      <p>Each button also sets DPP weight to the paper's choice (Quality for MAP, Importance for Samp) and turns the lock on, so the drawn vectors are the ones the method scores. A button stays pressed while the picking mode, $T'$ and $K$ match, $T$ is one of its grid values, and the DPP weight is the preset's. If you unlock the vector length and draw something else, the note under the slider says which vectors are shown. The chips jump to the paper's values of $T$.</p>`,
  },
  cone: {
    title: 'Open the cone',
    body: `<p>Completions of one prompt have embeddings that point in nearly the same direction, so the points sit in a narrow cone around their mean direction. Opening the cone stretches every direction perpendicular to that mean by a factor $s$ and leaves the mean direction alone. The factor is the ratio of the points' spread along the mean direction to their spread across it, between 1 and 4.</p>
      <p>The stretch is one linear map, so the box spanned by the selected arrows stays a box. Every 3D volume is multiplied by $s^2$ (every area by $s$), so the ratios between volumes are unchanged. The DPP itself is unchanged; only the picture differs.</p>`,
  },
};

let openBtn = null;

function typeset(el) {
  window.renderMathInElement?.(el, {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
    throwOnError: false,
  });
}

export function closeExplainer() {
  const pop = document.getElementById('xpop');
  pop.hidden = true;
  openBtn?.setAttribute('aria-expanded', 'false');
  openBtn = null;
}

function openExplainer(btn) {
  const item = EXPLAIN[btn.dataset.x];
  if (!item) return;
  const pop = document.getElementById('xpop');
  pop.innerHTML = `<div class="xp-title">${item.title}</div>${typeof item.body === 'function' ? item.body() : item.body}<button type="button" class="xp-close" aria-label="Close">×</button>`;
  typeset(pop);
  pop.querySelector('.xp-close').onclick = closeExplainer;
  pop.hidden = false;
  openBtn?.setAttribute('aria-expanded', 'false');
  openBtn = btn;
  btn.setAttribute('aria-expanded', 'true');
  // Place below the button (above if there is no room), clamped to the viewport.
  const r = btn.getBoundingClientRect();
  const w = Math.min(360, window.innerWidth - 24);
  pop.style.width = w + 'px';
  const left = Math.max(12, Math.min(window.innerWidth - w - 12, r.left + r.width / 2 - w / 2));
  const h = pop.offsetHeight;
  let top = r.bottom + 8;
  if (top + h > window.innerHeight - 8 && r.top - h - 8 > 8) top = r.top - h - 8;
  pop.style.left = left + window.scrollX + 'px';
  pop.style.top = Math.max(8, top) + window.scrollY + 'px';
}

export function initExplainers() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.xbtn');
    if (btn) {
      e.preventDefault(); // don't toggle the checkbox or focus the input of a surrounding <label>
      e.stopPropagation();
      if (openBtn === btn) closeExplainer(); else openExplainer(btn);
      return;
    }
    if (!e.target.closest('#xpop')) closeExplainer();
  }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeExplainer(); });
  window.addEventListener('resize', closeExplainer);
}
