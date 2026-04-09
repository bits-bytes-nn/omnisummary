export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const targetUrl = url.searchParams.get("url");

    if (!targetUrl) {
      return new Response("omnisummary-proxy", { status: 200 });
    }

    const authToken = url.searchParams.get("token");
    if (authToken !== env.PROXY_TOKEN) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }

    try {
      const headers = new Headers();
      headers.set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36");
      headers.set("Accept", request.headers.get("Accept") || "*/*");
      headers.set("Accept-Language", "en-US,en;q=0.9");

      const customHeaders = url.searchParams.get("headers");
      if (customHeaders) {
        const parsed = JSON.parse(customHeaders);
        for (const [key, value] of Object.entries(parsed)) {
          headers.set(key, value);
        }
      }

      const response = await fetch(targetUrl, {
        method: request.method,
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
      return new Response(JSON.stringify({ error: error.message }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }
  },
};
