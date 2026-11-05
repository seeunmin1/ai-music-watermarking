async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}.`);
  return response.json();
}

export async function checkBackendHealth(baseUrl) {
  const health = await fetchJson(`${baseUrl.replace(/\/$/, "")}/health`);
  if (health?.ok !== true) throw new Error("Invalid backend health response.");
  return health;
}

export async function detectWithBackend(baseUrl, file) {
  const body = new FormData();
  body.append("file", file);
  const result = await fetchJson(`${baseUrl.replace(/\/$/, "")}/detect`, { method: "POST", body });
  if (!result?.detection || typeof result?.provenance?.generationClaimLevel !== "string"
    || typeof result.provenance.displayTitle !== "string") {
    throw new Error("Invalid backend detection response.");
  }
  return result;
}
