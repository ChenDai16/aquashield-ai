/* AquaShield AI – service worker: lưu toàn bộ ứng dụng vào máy để chạy khi không có mạng.
   Khi thay tệp trên máy chủ: mở ứng dụng lúc có mạng → mục "Thông tin" → "Xóa bộ nhớ đệm và tải lại". */
const CACHE = 'aquashield-v2';
const CORE = [
  './', 'index.html', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon.png', 'icons/logo-96.png', 'icons/qr.png',
  'samples/hs.jpg', 'samples/bg.jpg', 'samples/wssv.jpg', 'samples/bg_wssv.jpg',
  'samples/hs_t.jpg', 'samples/bg_t.jpg', 'samples/wssv_t.jpg', 'samples/bg_wssv_t.jpg',
  'ort/ort.wasm.min.js', 'ort/ort-wasm-simd-threaded.js', 'ort/ort-wasm-simd-threaded.wasm'
];

self.addEventListener('install', (e) => {
  // cache:'reload' = luôn lấy tệp mới nhất từ máy chủ, không dùng bản trình duyệt còn giữ tạm
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE.map((u) => new Request(u, { cache: 'reload' })))).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const old = (await caches.keys()).filter((k) => k !== CACHE);
    await Promise.all(old.map((k) => caches.delete(k)));
    await self.clients.claim();
    // Máy đang giữ bản cũ: tự tải lại trang một lần để hiện ngay giao diện mới (có nút "Cài đặt ứng dụng").
    if (old.some((k) => k.indexOf('aquashield-') === 0)) {
      const wins = await self.clients.matchAll({ type: 'window' });
      wins.forEach((w) => { w.navigate(w.url).catch(() => {}); });   // không "await": chờ ở đây sẽ làm trang treo
    }
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return hit;                                   // có trong máy → dùng ngay, không cần mạng
    try {
      const res = await fetch(req);
      if (res && res.status === 200 && res.type === 'basic') cache.put(req, res.clone());
      return res;
    } catch (err) {
      if (req.mode === 'navigate') { const idx = await cache.match('index.html'); if (idx) return idx; }
      throw err;
    }
  })());
});
