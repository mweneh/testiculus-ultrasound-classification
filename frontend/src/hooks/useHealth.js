import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchHealth } from '../lib/api.js';

/**
 * Poll GET /health. Model availability is exactly the set of models the
 * running service reports as loaded; anything else cannot run.
 */
export function useHealth({ intervalMs = 30_000 } = {}) {
  const [state, setState] = useState({ phase: 'checking', models: {}, error: null, checkedAt: null });
  const controllerRef = useRef(null);

  const refresh = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      const health = await fetchHealth({ signal: controller.signal });
      setState({ phase: 'online', models: health.models, error: null, checkedAt: new Date() });
    } catch (error) {
      if (error?.name === 'AbortError') return;
      setState({ phase: 'offline', models: {}, error, checkedAt: new Date() });
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, intervalMs);
    const onVisible = () => {
      if (document.visibilityState === 'visible') refresh();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
      controllerRef.current?.abort();
    };
  }, [refresh, intervalMs]);

  return { ...state, refresh };
}
