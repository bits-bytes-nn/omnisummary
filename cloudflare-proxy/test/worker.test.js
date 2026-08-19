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

/**
 * Replace global fetch with a router keyed by URL: a string value is a 200 body, a `{ location }`
 * value is a 302 to that Location. Redirects are what the allowlist has to survive.
 */
const withRoutedFetch = async (routes, run) => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    calls.push({ url, init });
    const route = routes[url];
    if (route === undefined) throw new Error(`unrouted fetch: ${url}`);
    if (typeof route === "object" && route.location !== undefined) {
      return new Response(null, { status: route.status ?? 302, headers: { Location: route.location } });
    }
    return new Response(route, { status: 200, headers: { "Content-Type": "text/xml" } });
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

test("a redirect is followed only while it stays on an allowed host", async () => {
  const start = "https://old.reddit.com/r/x/.rss";
  const hop = "https://www.reddit.com/r/x/.rss";
  await withRoutedFetch({ [start]: { location: hop }, [hop]: "<rss/>" }, async (calls) => {
    const response = await worker.fetch(proxyRequest(start), ENV);
    assert.equal(response.status, 200);
    assert.equal(await response.text(), "<rss/>");
    assert.deepEqual(
      calls.map((c) => c.url),
      [start, hop],
    );
    // redirect: "follow" would let the RUNTIME chase the Location without any allowlist check.
    assert.equal(calls[0].init.redirect, "manual");
  });
});

test("a redirect off the allowlist is refused instead of quietly proxied", async () => {
  // THE HOLE: isAllowedHost only ever saw the `url` parameter, and fetch followed redirects itself,
  // so one 302 from reddit/youtube turned this authenticated worker into an open proxy.
  const start = "https://old.reddit.com/r/x/.rss";
  for (const evil of ["https://evil.example/pwn", "http://169.254.169.254/latest/meta-data/"]) {
    await withRoutedFetch({ [start]: { location: evil } }, async (calls) => {
      const response = await worker.fetch(proxyRequest(start), ENV);
      assert.equal(response.status, 403, evil);
      assert.match((await response.json()).error, /Redirect target host not allowed/);
      assert.equal(calls.length, 1, "the off-allowlist hop must never be fetched");
    });
  }
});

test("a relative redirect resolves against the current hop and stays allowed", async () => {
  const start = "https://old.reddit.com/r/x/.rss";
  const hop = "https://old.reddit.com/r/y/.rss";
  await withRoutedFetch({ [start]: { location: "/r/y/.rss" }, [hop]: "<rss/>" }, async (calls) => {
    const response = await worker.fetch(proxyRequest(start), ENV);
    assert.equal(response.status, 200);
    assert.equal(calls[1].url, hop);
  });
});

test("a non-http redirect scheme is refused", async () => {
  const start = "https://old.reddit.com/r/x/.rss";
  await withRoutedFetch({ [start]: { location: "file:///etc/passwd" } }, async () => {
    const response = await worker.fetch(proxyRequest(start), ENV);
    assert.equal(response.status, 403);
    assert.match((await response.json()).error, /Unsupported redirect scheme/);
  });
});

test("a redirect loop is bounded rather than spinning", async () => {
  const a = "https://old.reddit.com/a";
  const b = "https://old.reddit.com/b";
  await withRoutedFetch({ [a]: { location: b }, [b]: { location: a } }, async (calls) => {
    const response = await worker.fetch(proxyRequest(a), ENV);
    assert.equal(response.status, 502);
    assert.match((await response.json()).error, /Too many redirects/);
    assert.ok(calls.length <= 5, `bounded hop count, got ${calls.length}`);
  });
});

test("a 3xx with no Location is passed through, not treated as a redirect", async () => {
  const target = "https://old.reddit.com/r/x/.rss";
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(null, { status: 304 });
  try {
    const response = await worker.fetch(proxyRequest(target), ENV);
    assert.equal(response.status, 304);
  } finally {
    globalThis.fetch = original;
  }
});

test("the outbound User-Agent comes from the USER_AGENT var", async () => {
  await withStubbedFetch("<rss/>", async (calls) => {
    await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss"), { ...ENV, USER_AGENT: "Mozilla/5.0 (configured)" });
    assert.equal(new Headers(calls[0].init.headers).get("User-Agent"), "Mozilla/5.0 (configured)");
  });
  // Fallback keeps the worker deployable if the var is ever dropped.
  await withStubbedFetch("<rss/>", async (calls) => {
    await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss"), ENV);
    assert.match(new Headers(calls[0].init.headers).get("User-Agent"), /Mozilla/);
  });
});

test("a token differing only in length is still rejected", async () => {
  // The compare is constant-time, so it must not accept a prefix (or a longer string) either.
  for (const token of ["secret-toke", "secret-tokenn", ""]) {
    const response = await worker.fetch(proxyRequest("https://old.reddit.com/r/x/.rss", { token }), ENV);
    assert.equal(response.status, 401, token);
  }
});
