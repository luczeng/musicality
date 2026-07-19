const CACHE_NAME = "musicality-mobile-companion-v11";
const APP_SHELL = [
  "/",
  "/manifest.json",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/app.js",
  "/static/queue.js",
];

// cache.addAll() is atomic — one failed fetch (e.g. the server restarting
// mid-request, as happened during development) rejects the whole install,
// which leaves the *old* service worker permanently in control with no
// visible error, since a rejected `install` never reaches `activate` (where
// stale caches normally get cleaned up). Promise.allSettled instead lets
// the install succeed with whatever it could fetch — anything that failed
// just falls through to a normal network request at runtime — and logs
// failures so they're visible in the console instead of silently stuck.
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      const results = await Promise.allSettled(
        APP_SHELL.map(async (url) => {
          const response = await fetch(url, { cache: "no-store" });
          if (!response.ok) throw new Error(`${url}: ${response.status}`);
          await cache.put(url, response);
        })
      );
      const failed = results
        .map((r, i) => (r.status === "rejected" ? APP_SHELL[i] : null))
        .filter(Boolean);
      if (failed.length > 0) {
        console.warn("[sw] failed to precache (will retry from network):", failed);
      } else {
        console.log("[sw] precached app shell:", CACHE_NAME);
      }
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) =>
        Promise.all(
          names
            .filter((name) => name !== CACHE_NAME)
            .map((name) => {
              console.log("[sw] deleting stale cache:", name);
              return caches.delete(name);
            })
        )
      )
  );
  self.clients.claim();
  console.log("[sw] activated:", CACHE_NAME);
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
