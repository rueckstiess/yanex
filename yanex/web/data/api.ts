/** Shared HTTP boundary. Callers own cancellation and response types. */
export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch { /* Non-JSON error responses retain the HTTP status. */ }
    throw new Error(message)
  }
  return response.json()
}
