// sw.js
self.addEventListener("install", event => {
    event.waitUntil(
      caches.open("emotion-tracker-v1").then(cache => {
        return cache.addAll([
          "./",
          "./index.html",
          "./manifest.json"
          // 以后如果把 Vue、Chart.js 换成本地文件，也可以加进来
        ])
      })
    )
  })
  
  self.addEventListener("fetch", event => {
    event.respondWith(
      caches.match(event.request).then(resp => {
        return resp || fetch(event.request)
      })
    )
  })
  