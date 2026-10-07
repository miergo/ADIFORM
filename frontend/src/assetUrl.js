/** Public-folder path, prefixed with Vite `base` (needed on GitHub Pages). */
export function assetUrl(path) {
  const base = import.meta.env.BASE_URL ?? "/";
  const clean = String(path).replace(/^\/+/, "");
  return `${base}${clean}`;
}
