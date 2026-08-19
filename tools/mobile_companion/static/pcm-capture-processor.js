/**
 * Batches mono mic PCM into ~4096-sample chunks (~93ms @44.1kHz) before
 * posting to the main thread — cuts message volume from ~344/s (one per
 * 128-sample render quantum) to ~11/s. Each chunk's buffer is transferred
 * (zero-copy) since `.slice()` gives it a fresh backing ArrayBuffer.
 */
class PCMCaptureProcessor extends AudioWorkletProcessor {
  static CHUNK_SIZE = 4096;

  constructor() {
    super();

    this._buffer = new Float32Array(PCMCaptureProcessor.CHUNK_SIZE);
    this._writeIndex = 0;

    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") this._flush();
    };
  }

  // `disconnect()` only stops future process() calls — it can't force-emit
  // a partial buffer, so without this the tail (up to one chunk) would be
  // silently dropped on every recording. The main thread awaits "flushed"
  // before finalizing; MessagePort delivers in order, so any trailing
  // "chunk" is guaranteed to arrive before it.
  _flush() {
    if (this._writeIndex > 0) {
      const chunk = this._buffer.slice(0, this._writeIndex);
      this.port.postMessage({ type: "chunk", samples: chunk }, [chunk.buffer]);
      this._writeIndex = 0;
    }
    this.port.postMessage({ type: "flushed" });
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    let read = 0;
    while (read < channel.length) {
      const spaceLeft = PCMCaptureProcessor.CHUNK_SIZE - this._writeIndex;
      const toCopy = Math.min(spaceLeft, channel.length - read);
      this._buffer.set(channel.subarray(read, read + toCopy), this._writeIndex);
      this._writeIndex += toCopy;
      read += toCopy;

      if (this._writeIndex === PCMCaptureProcessor.CHUNK_SIZE) {
        const chunk = this._buffer.slice(0, this._writeIndex);
        this.port.postMessage({ type: "chunk", samples: chunk }, [chunk.buffer]);
        this._writeIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
