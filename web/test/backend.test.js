import assert from "node:assert/strict";
import test from "node:test";
import { checkBackendHealth, detectWithBackend } from "../src/backend.js";

const file = new Blob(["audio"], { type: "audio/mpeg" });

test("health checks require the backend's JSON acknowledgement", async (t) => {
  t.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, "/api/health");
    return Response.json({ ok: true });
  });
  assert.deepEqual(await checkBackendHealth("/api/"), { ok: true });
});

test("HTML fallback pages cannot be mistaken for a healthy backend", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("<!doctype html>"));
  await assert.rejects(checkBackendHealth("/api"));
});

test("unhealthy JSON responses are rejected", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json({ ok: false }));
  await assert.rejects(checkBackendHealth("/api"), /Invalid backend health/);
});

test("detection HTTP errors trigger fallback instead of silently succeeding", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("error", { status: 413 }));
  await assert.rejects(detectWithBackend("/api", file), /HTTP 413/);
});

test("network errors propagate to the visible fallback handler", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new Error("offline"); });
  await assert.rejects(detectWithBackend("/api", file), /offline/);
});

test("invalid detection JSON is rejected", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json({}));
  await assert.rejects(detectWithBackend("/api", file), /Invalid backend detection/);
});

test("non-JSON detection responses are rejected", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("<!doctype html>"));
  await assert.rejects(detectWithBackend("/api", file));
});

test("backend declarations retain their untrusted status and upload uses multipart", async (t) => {
  const result = {
    detection: { found: false },
    provenance: {
      generationClaimLevel: "detected",
      providerVerificationStatus: "not_verified",
      displayTitle: "AI-generated (declared, signer not trusted)",
    },
  };
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "http://localhost:8787/detect");
    assert.equal(options.method, "POST");
    assert.ok(options.body instanceof FormData);
    assert.equal(options.body.get("file").size, file.size);
    return Response.json(result);
  });
  assert.deepEqual(await detectWithBackend("http://localhost:8787/", file), result);
});
