// Hand-rolled cache-first service worker. No deps.
// Cache version bumps whenever the cached set changes meaningfully.
const CACHE = 'easy-yahtzee-v1';
const PRECACHE = ['/', '/scores.bin'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  event.respondWith(
    caches.match(request).then((hit) => {
      if (hit) return hit;
      return fetch(request).then((resp) => {
        // Cache same-origin successful responses for next time.
        if (resp.ok && new URL(request.url).origin === self.location.origin) {
          const clone = resp.clone();
          caches.open(CACHE).then((c) => c.put(request, clone));
        }
        return resp;
      });
    }),
  );
});
