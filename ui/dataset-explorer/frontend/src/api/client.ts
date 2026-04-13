import { QueryClient } from "@tanstack/react-query";

// Base URL — Vite proxies /api → localhost:8000 in dev.
// In production, set VITE_API_URL to your Railway backend URL.
export const API_BASE = import.meta.env.VITE_API_URL ?? "";
const API_FETCH_TIMEOUT_MS = 60_000;

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000, // 30s before refetching
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * Typed fetch wrapper. Throws on non-2xx responses.
 */
export async function apiFetch<T>(
  path: string,
  params?: Record<string, string | number | boolean | undefined>,
): Promise<T> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);

  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    }
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), API_FETCH_TIMEOUT_MS);

  try {
    const res = await fetch(url.toString(), { signal: controller.signal });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API error ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(`API request timed out after ${API_FETCH_TIMEOUT_MS}ms`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}
