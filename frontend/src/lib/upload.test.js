import { describe, expect, it } from 'vitest';
import { MAX_UPLOAD_BYTES, sniffFormat, validateImageFile } from './upload.js';

const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
const JPEG = [0xff, 0xd8, 0xff, 0xe0, 0, 16];
const GIF = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61];

const file = (bytes, name = 'image.png', type = 'image/png') => new File([new Uint8Array(bytes)], name, { type });
const size = (width, height) => async () => ({ width, height });

describe('sniffFormat', () => {
  it('identifies PNG and JPEG by signature, not by extension', () => {
    expect(sniffFormat(new Uint8Array(PNG).buffer)).toBe('PNG');
    expect(sniffFormat(new Uint8Array(JPEG).buffer)).toBe('JPEG');
    expect(sniffFormat(new Uint8Array(GIF).buffer)).toBeNull();
  });
});

describe('validateImageFile', () => {
  it('accepts a decodable PNG within limits without uploading it', async () => {
    await expect(validateImageFile(file(PNG), { dimensions: size(300, 300) })).resolves.toEqual({
      ok: true,
      format: 'PNG',
      width: 300,
      height: 300,
    });
  });

  it('accepts JPEG content', async () => {
    const result = await validateImageFile(file(JPEG, 'scan.jpg', 'image/jpeg'), { dimensions: size(640, 480) });
    expect(result).toMatchObject({ ok: true, format: 'JPEG' });
  });

  it('rejects empty files, unsupported formats and renamed files', async () => {
    expect(await validateImageFile(file([]), { dimensions: size(1, 1) })).toMatchObject({ ok: false, error: 'This file is empty.' });
    expect(await validateImageFile(file(GIF, 'x.gif', 'image/gif'), { dimensions: size(1, 1) })).toMatchObject({
      ok: false,
      error: 'Only PNG and JPEG images are supported.',
    });
    expect(await validateImageFile(file(GIF, 'renamed.png'), { dimensions: size(1, 1) })).toMatchObject({ ok: false });
  });

  it('mirrors the server 5 MiB and 20 megapixel limits', async () => {
    const big = file(PNG);
    Object.defineProperty(big, 'size', { value: MAX_UPLOAD_BYTES + 1 });
    expect((await validateImageFile(big, { dimensions: size(1, 1) })).error).toMatch(/limit is 5 MiB/);
    expect((await validateImageFile(file(PNG), { dimensions: size(5000, 4001) })).error).toMatch(/20 megapixel/);
    expect((await validateImageFile(file(PNG), { dimensions: size(5000, 4000) })).ok).toBe(true);
  });

  it('reports undecodable images', async () => {
    const broken = async () => {
      throw new Error('undecodable');
    };
    expect((await validateImageFile(file(PNG), { dimensions: broken })).error).toMatch(/could not be decoded/);
  });
});
