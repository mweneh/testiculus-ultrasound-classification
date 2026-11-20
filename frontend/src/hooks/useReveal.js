import { useEffect, useRef, useState } from 'react';

function revealImmediately() {
  if (typeof window === 'undefined' || !('IntersectionObserver' in window)) return true;
  return Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
}

/**
 * Reports once whether the referenced element has scrolled into view. Without
 * IntersectionObserver support, or when the viewer prefers reduced motion,
 * content counts as revealed from the first render so nothing stays hidden.
 */
export function useReveal({ threshold = 0.15 } = {}) {
  const ref = useRef(null);
  const [revealed, setRevealed] = useState(revealImmediately);

  useEffect(() => {
    if (revealed || !ref.current) return undefined;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setRevealed(true);
        observer.disconnect();
      },
      { threshold, rootMargin: '0px 0px -8% 0px' },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [revealed, threshold]);

  return [ref, revealed];
}
