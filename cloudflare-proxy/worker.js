// Fetch proxy for the collectors: some sources (Reddit, YouTube) block AWS datacenter IPs but
// allow this worker's. It is deliberately NOT a general-purpose proxy — the target host must be
// on ALLOWED_HOSTS and the caller must present PROXY_TOKEN (a wrangler secret, never in the repo).
const jsonResponse = (body, status) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

// Exact or suffix match, so "reddit.com" covers www.reddit.com / old.reddit.com without listing
// every subdomain, while "evil-reddit.com" (a mere substring match) stays rejected.
const isAllowedHost = (host, allowedHosts) =>
  allowedHosts.some((allowed) => host === allowed || host.endsWith(`.${allowed}`));

// Compare the token without an early exit, so response time cannot leak it byte by byte. Written by
// hand because crypto.subtle.timingSafeEqual is a Workers-only extension and absent under node:test.
const timingSafeEqual = (a, b) => {
  const encoder = new TextEncoder();
  const left = encoder.encode(a);
  const right = encoder.encode(b);
  let mismatch = left.length ^ right.length;
  const length = Math.max(left.length, right.length);
  for (let i = 0; i < length; i++) {
    mismatch |= (left[i] ?? 0) ^ (right[i] ?? 0);
  }
  return mismatch === 0;
};

const isHttpScheme = (url) => url.protocol === "https:" || url.protocol === "http:";

const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

// Redirect hops followed before giving up. Bounded because every hop is a fresh outbound fetch on
// this worker's IP, and a redirect loop would otherwise spin until the runtime kills the request.
const MAX_REDIRECTS = 3;

// Fallback only: the live value is the USER_AGENT var in wrangler.toml, kept in step with
// shared.constants.BROWSER_USER_AGENT (the two copies had already drifted a Chrome major apart).
const DEFAULT_USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const targetUrl = url.searchParams.get("url");

    if (!targetUrl) {
      return new Response("omnisummary-proxy", { status: 200 });
    }

    // Token stays a QUERY parameter: both live callers hand the proxied URL straight to a plain GET
    // whose headers they do not control.
    if (!env.PROXY_TOKEN || !timingSafeEqual(url.searchParams.get("token") || "", env.PROXY_TOKEN)) {
      return jsonResponse({ error: "Unauthorized" }, 401);
    }

    let target;
    try {
      target = new URL(targetUrl);
    } catch {
      return jsonResponse({ error: "Invalid url parameter" }, 400);
    }
    if (!isHttpScheme(target)) {
      return jsonResponse({ error: "Unsupported scheme" }, 400);
    }

    const allowedHosts = (env.ALLOWED_HOSTS || "")
      .split(",")
      .map((h) => h.trim().toLowerCase())
      .filter(Boolean);
    if (!isAllowedHost(target.hostname.toLowerCase(), allowedHosts)) {
      return jsonResponse({ error: "Target host not allowed" }, 403);
    }

    try {
      // Fixed request headers only. A caller-supplied `headers` JSON blob used to be merged in
      // verbatim, which let anyone holding the token forge Cookie/Authorization/Host on the
      // outbound request.
      const headers = new Headers({
        "User-Agent": env.USER_AGENT || DEFAULT_USER_AGENT,
        Accept: "*/*",
        "Accept-Language": "en-US,en;q=0.9",
      });
      const method = request.method === "HEAD" ? "HEAD" : "GET";

      let current = target;
      let response;
      for (let hop = 0; ; hop++) {
        // MANUAL, not "follow": with the runtime following redirects, a single 302 from an allowed
        // host (reddit/youtube) sent this authenticated worker to ANY host — the open proxy the
        // allowlist exists to prevent. Each hop is re-checked against the same allowlist below.
        response = await fetch(current.toString(), { method, headers, redirect: "manual" });

        const location = REDIRECT_STATUSES.has(response.status) ? response.headers.get("Location") : null;
        if (!location) break;
        if (hop >= MAX_REDIRECTS) {
          return jsonResponse({ error: "Too many redirects" }, 502);
        }

        let next;
        try {
          next = new URL(location, current);
        } catch {
          return jsonResponse({ error: "Invalid redirect location" }, 502);
        }
        if (!isHttpScheme(next)) {
          return jsonResponse({ error: "Unsupported redirect scheme" }, 403);
        }
        if (!isAllowedHost(next.hostname.toLowerCase(), allowedHosts)) {
          return jsonResponse({ error: "Redirect target host not allowed" }, 403);
        }
        current = next;
      }

      const responseHeaders = new Headers();
      responseHeaders.set("Content-Type", response.headers.get("Content-Type") || "application/octet-stream");
      responseHeaders.set("Access-Control-Allow-Origin", "*");

      return new Response(response.body, {
        status: response.status,
        headers: responseHeaders,
      });
    } catch (error) {
      return jsonResponse({ error: error.message }, 502);
    }
  },
};
