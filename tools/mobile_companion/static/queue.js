/**
 * IndexedDB-backed queue of not-yet-synced captures (recording blob + tap
 * times). Writing here never touches the network — this is what lets
 * recording work with zero signal. The sync engine (step 10) later drains
 * this queue into the backend endpoints from steps 5–6.
 */

const DB_NAME = "musicality-mobile-companion";
const DB_VERSION = 1;
const STORE_NAME = "captures";

let dbPromise = null;

function openDB() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        request.result.createObjectStore(STORE_NAME, { keyPath: "id" });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  return dbPromise;
}

function promisifyRequest(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// A transaction auto-commits once no request is pending and control returns
// to the event loop — fine across microtask (Promise) boundaries like the
// `await openDB()` below, but it would silently break if any *macrotask*
// work (e.g. a `fetch`) were awaited between opening the store and issuing
// the request. Keep these functions free of network calls for that reason.
async function getStore(mode) {
  const db = await openDB();
  return db.transaction(STORE_NAME, mode).objectStore(STORE_NAME);
}

/**
 * Save a captured recording + tap times locally. Returns the generated id.
 */
export async function addPendingCapture(
  blob,
  tapTimes,
  dataset,
  trackName,
  structure,
  device
) {
  const store = await getStore("readwrite");
  const id = crypto.randomUUID();
  await promisifyRequest(
    store.add({
      id,
      dataset,
      trackName: trackName || null,
      tapTimes,
      structure: structure || null,
      device: device || null,
      blob,
      createdAt: Date.now(),
      synced: false,
    })
  );
  return id;
}

/**
 * Return every capture not yet synced, oldest first.
 */
export async function listPending() {
  const store = await getStore("readonly");
  const all = await promisifyRequest(store.getAll());
  return all
    .filter((capture) => !capture.synced)
    .sort((a, b) => a.createdAt - b.createdAt);
}

/**
 * Flag a capture as synced without removing it (kept as a record until the
 * caller also calls deletePending, if it wants the queue entry gone too).
 */
export async function markSynced(id) {
  const store = await getStore("readwrite");
  const capture = await promisifyRequest(store.get(id));
  if (!capture) return;
  capture.synced = true;
  await promisifyRequest(store.put(capture));
}

/**
 * Remove a capture from the queue entirely.
 */
export async function deletePending(id) {
  const store = await getStore("readwrite");
  await promisifyRequest(store.delete(id));
}

/**
 * Discard every capture in the queue, synced or not. For manually clearing
 * out stuck/unwanted captures — irreversible, callers should confirm first.
 */
export async function flushQueue() {
  const store = await getStore("readwrite");
  await promisifyRequest(store.clear());
}
