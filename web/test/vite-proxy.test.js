import assert from "node:assert/strict";
import { createServer as createHttpServer } from "node:http";
import { once } from "node:events";
import test from "node:test";
import { createServer } from "vite";
import config from "../vite.config.js";
import { checkBackendHealth, detectWithBackend } from "../src/backend.js";

test("Vite routes the frontend API requests to the Python-style backend paths", async (t) => {
  const paths = [];
  const backend = createHttpServer((request, response) => {
    paths.push([request.method, request.url]);
    request.resume();
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify(request.url === "/health" ? { ok: true } : {
      detection: { found: false },
      provenance: { generationClaimLevel: "detected", displayTitle: "C2PA manifest found" },
    }));
  });
  backend.listen(0, "127.0.0.1");
  await once(backend, "listening");
  t.after(() => new Promise((resolve) => backend.close(resolve)));
  assert.equal(config.server.proxy["/api"].target, "http://127.0.0.1:8787");
  const vite = await createServer({
    ...config,
    configFile: false,
    logLevel: "silent",
    server: {
      host: "127.0.0.1", port: 0,
      proxy: {
        "/api": { ...config.server.proxy["/api"], target: `http://127.0.0.1:${backend.address().port}` },
      },
    },
  });
  t.after(() => vite.close());
  await vite.listen();
  const baseUrl = `http://127.0.0.1:${vite.httpServer.address().port}/api`;
  await checkBackendHealth(baseUrl);
  const result = await detectWithBackend(baseUrl, new Blob(["audio"]));
  assert.equal(result.provenance.generationClaimLevel, "detected");
  assert.deepEqual(paths, [["GET", "/health"], ["POST", "/detect"]]);
});
