// Executable tests for the worker's request handling. The Python side (tests/test_proxy.py) can
// only assert that worker.js CONTAINS a string, which stays green even if isAllowedHost returned
// true unconditionally or the 401/403 branches were inverted — and it re-implements the suffix
// match in the test, so it can drift from the code it claims to check. This is the only
// internet-reachable component in the repo, so its five branches (token, URL parse, scheme,
// suffix host match, fixed outbound headers) are exercised for real here.
//
// node:test + node:assert only: no new dependency, and CI already installs Node for the CDK synth.
import assert from "node:assert/strict";
import test from "node:test";

import worker from "../worker.js";

const ENV = { PROXY_TOKEN: "secret-token", ALLOWED_HOSTS: "reddit.com,youtube.com,youtu.be" };

const PROXY_URL = "https://proxy.example.workers.dev/";

/** Build a request to the worker for `target`, with an optional token/extra query params. */
const proxyRequest = (target, { token = ENV.PROXY_TOKEN, extra = {}, method = "GET" } = {}) => {
  const url = new URL(PROXY_URL);
  if (target !== undefined) url.searchParams.set("url", target);
  if (token !== null) url.searchParams.set("token", token);
  for (const [key, value] of Object.entries(extra)) url.searchParams.set(key, value);
  return new Request(url, { method });
};

/** Replace global fetch with a recorder, returning the calls it saw. Always restored. */
const withStubbedFetch = async (body, run) => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (input, init) => {
    calls.push({ url: String(input), init });
    return new Response(body, { status: 200, headers: { "Content-Type": "text/xml" } });
  };
  try {
    await run(calls);
  } finally {
    globalThis.fetch = original;
  }
};

test("a request with no url parameter is the health probe, not a proxy attempt", async () => {
  const response = await worker.fetch(new Request(PROXY_URL), ENV);
  assert.equal(response.status, 200);
  assert.equal(await response.text(), "omnisummary-proxy");
});

test("a missing token is rejected", async () => {
  const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { token: null }), ENV);
  assert.equal(response.status, 401);
});

test("a wrong token is rejected", async () => {
  const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { token: "nope" }), ENV);
  assert.equal(response.status, 401);
});

test("an unset PROXY_TOKEN rejects everything instead of opening the proxy", async () => {
  const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { token: "" }), {
    ...ENV,
    PROXY_TOKEN: "",
  });
  assert.equal(response.status, 401);
});

test("an unparseable url parameter is a 400", async () => {
  const response = await worker.fetch(proxyRequest("not-a-url"), ENV);
  assert.equal(response.status, 400);
});

test("a non-http(s) scheme is refused", async () => {
  for (const target of ["ftp://reddit.com/x", "file:///etc/passwd"]) {
    const response = await worker.fetch(proxyRequest(target), ENV);
    assert.equal(response.status, 400, target);
  }
});

test("a host that merely CONTAINS an allowed domain is refused", async () => {
  for (const target of [
    "https://evil-reddit.com/r/x/.rss",
    "https://reddit.com.evil.example/r/x/.rss",
    "http://169.254.169.254/latest/meta-data/",
  ]) {
    const response = await worker.fetch(proxyRequest(target), ENV);
    assert.equal(response.status, 403, target);
  }
});

test("an allowed host and its subdomains are proxied", async () => {
  for (const target of ["https://reddit.com/r/x/.rss", "https://old.reddit.com/r/x/.rss"]) {
    await withStubbedFetch("<rss/>", async (calls) => {
      const response = await worker.fetch(proxyRequest(target), ENV);
      assert.equal(response.status, 200, target);
      assert.equal(await response.text(), "<rss/>");
      assert.equal(calls.length, 1);
      assert.equal(calls[0].url, target);
    });
  }
});

test("caller-supplied headers never reach the outbound request", async () => {
  const extra = {
    headers: JSON.stringify({ Cookie: "session=stolen", Authorization: "Bearer forged", Host: "internal" }),
  };
  await withStubbedFetch("<rss/>", async (calls) => {
    const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { extra }), ENV);
    assert.equal(response.status, 200);
    const sent = new Headers(calls[0].init.headers);
    assert.equal(sent.get("Cookie"), null);
    assert.equal(sent.get("Authorization"), null);
    assert.equal(sent.get("Host"), null);
    assert.match(sent.get("User-Agent"), /Mozilla/);
  });
});

test("only GET and HEAD are forwarded, whatever the caller's method", async () => {
  await withStubbedFetch("<rss/>", async (calls) => {
    await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { method: "HEAD" }), ENV);
    assert.equal(calls[0].init.method, "HEAD");
  });
  await withStubbedFetch("<rss/>", async (calls) => {
    await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { method: "DELETE" }), ENV);
    assert.equal(calls[0].init.method, "GET");
  });
});

test("an empty ALLOWED_HOSTS allows nothing", async () => {
  const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss"), { ...ENV, ALLOWED_HOSTS: "" });
  assert.equal(response.status, 403);
});

test("an upstream fetch failure surfaces as 502, not a crash", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("connection reset");
  };
  try {
    const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss"), ENV);
    assert.equal(response.status, 502);
    assert.equal((await response.json()).error, "connection reset");
  } finally {
    globalThis.fetch = original;
  }
});
