import { useEffect, useRef } from 'react';

/** Same attribution colour as src/gradcam.make_overlay (red-orange, alpha ∝ map). */
export const HEAT_RGB = [255, 40, 0];

/**
 * Draw the server's 224×224 grayscale Grad-CAM map as a translucent layer.
 * The model resizes the whole image directly to 224×224 (no crop), so
 * stretching the map over the displayed image restores the original geometry.
 */
export default function HeatmapLayer({ pngBase64, opacity, visible }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext?.('2d');
    if (!context || !pngBase64) return undefined;
    let cancelled = false;
    const source = new Image();
    source.onload = () => {
      if (cancelled) return;
      canvas.width = source.naturalWidth;
      canvas.height = source.naturalHeight;
      context.drawImage(source, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const { data } = pixels;
      for (let index = 0; index < data.length; index += 4) {
        const intensity = data[index];
        data[index] = HEAT_RGB[0];
        data[index + 1] = HEAT_RGB[1];
        data[index + 2] = HEAT_RGB[2];
        data[index + 3] = intensity;
      }
      context.putImageData(pixels, 0, 0);
    };
    source.src = `data:image/png;base64,${pngBase64}`;
    return () => {
      cancelled = true;
    };
  }, [pngBase64]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-300"
      style={{ opacity: visible ? opacity : 0 }}
    />
  );
}
