// Pure helpers for the point labels (no DOM).

// Split text at the one word boundary that makes the two lines closest in length (the earlier one on a tie).
export function splitTwoLines(text) {
  const w = String(text).replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  if (w.length < 2) return [w.join(' ')];
  const total = w.join(' ').length;
  let best = 1, bestDiff = Infinity, left = 0;
  for (let k = 1; k < w.length; k++) {
    left += w[k - 1].length + (k > 1 ? 1 : 0);   // length of the first k words joined by spaces
    const diff = Math.abs(left - (total - left - 1));
    if (diff < bestDiff) { bestDiff = diff; best = k; }
  }
  return [w.slice(0, best).join(' '), w.slice(best).join(' ')];
}
