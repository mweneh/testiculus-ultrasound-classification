/** Client-side checks mirroring the server's upload limits in src/api.py. */

export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 20_000_000;

const SIGNATURES = [
  { format: 'PNG', bytes: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] },
  { format: 'JPEG', bytes: [0xff, 0xd8, 0xff] },
];

export function sniffFormat(header) {
  const bytes = new Uint8Array(header);
  const match = SIGNATURES.find(({ bytes: signature }) =>
    signature.every((value, index) => bytes[index] === value),
  );
  return match?.format ?? null;
}

async function readHeader(file) {
  const blob = file.slice(0, 16);
  if (typeof blob.arrayBuffer === 'function') return blob.arrayBuffer();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });
}

export function readImageDimensions(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('undecodable'));
    };
    image.src = url;
  });
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

/**
 * Validate a selected file before it is previewed. Nothing is uploaded here.
 * Returns { ok: true, format, width, height } or { ok: false, error }.
 */
export async function validateImageFile(file, { dimensions = readImageDimensions } = {}) {
  if (!file) return { ok: false, error: 'No file was selected.' };
  if (file.size === 0) return { ok: false, error: 'This file is empty.' };
  if (file.size > MAX_UPLOAD_BYTES) {
    return { ok: false, error: `This file is ${formatBytes(file.size)}; the limit is 5 MiB.` };
  }
  let format;
  try {
    format = sniffFormat(await readHeader(file));
  } catch {
    return { ok: false, error: 'The file could not be read.' };
  }
  if (!format) return { ok: false, error: 'Only PNG and JPEG images are supported.' };
  let size;
  try {
    size = await dimensions(file);
  } catch {
    return { ok: false, error: 'This image could not be decoded. It may be corrupted.' };
  }
  if (!size?.width || !size?.height) {
    return { ok: false, error: 'This image could not be decoded. It may be corrupted.' };
  }
  if (size.width * size.height > MAX_IMAGE_PIXELS) {
    return { ok: false, error: 'This image exceeds the 20 megapixel limit.' };
  }
  return { ok: true, format, width: size.width, height: size.height };
}
