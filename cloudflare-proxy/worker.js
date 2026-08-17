// Fetch proxy for the collectors: some sources (Reddit, YouTube) block AWS datacenter IPs but
// allow this worker's. It is deliberately NOT a general-purpose proxy — the target host must be
// on ALLOWED_HOSTS and the caller must present PROXY_TOKEN (a wrangler secret, never in the repo).
const jsonResponse = (body, status) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

// Exact or suffix match, so "reddit.com" covers www.reddit.com / old.reddit.com without listing
// every subdomain, while "evil-reddit.com" (a mere substring match) stays rejected.
const isAllowedHost = (host, allowedHosts) =>
  allowedHosts.some((allowed) => host === allowed || host.endsWith(`.${allowed}`));

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const targetUrl = url.searchParams.get("url");

    if (!targetUrl) {
      return new Response("omnisummary-proxy", { status: 200 });
    }

    // Token stays a QUERY parameter: both live callers hand the proxied URL straight to
    // feedparser.parse, which cannot attach headers.
    if (!env.PROXY_TOKEN || url.searchParams.get("token") !== env.PROXY_TOKEN) {
      return jsonResponse({ error: "Unauthorized" }, 401);
    }

    let target;
    try {
      target = new URL(targetUrl);
    } catch {
      return jsonResponse({ error: "Invalid url parameter" }, 400);
    }
    if (target.protocol !== "https:" && target.protocol !== "http:") {
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
        "User-Agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        Accept: "*/*",
        "Accept-Language": "en-US,en;q=0.9",
      });

      const response = await fetch(target.toString(), {
        method: request.method === "HEAD" ? "HEAD" : "GET",
        headers: headers,
        redirect: "follow",
      });

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
