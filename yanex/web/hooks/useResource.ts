import { useEffect, useState } from 'react'
import { getJson } from '../data/api'

/** Abort obsolete requests and prevent late responses from replacing current data. */
export function useResource<T>(url: string | null) {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true })
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (!url) return
    const controller = new AbortController()
    let active = true
    setState({ data: null, error: null, loading: true })
    getJson<T>(url, controller.signal).then(
      data => { if (active) setState({ data, error: null, loading: false }) },
      error => { if (active) setState({ data: null, error: error instanceof Error ? error.message : 'Request failed', loading: false }) },
    )
    return () => { active = false; controller.abort() }
  }, [url, revision])
  return { ...state, refresh: () => setRevision(value => value + 1) }
}
