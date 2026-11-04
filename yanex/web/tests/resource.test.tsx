import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { useResource } from '../hooks/useResource'
import { getJson } from '../data/api'

afterEach(() => vi.unstubAllGlobals())

test('obsolete responses cannot replace the current resource', async () => {
  const pending: Array<(value: unknown) => void> = []
  const fetcher = vi.fn(() => new Promise(resolve => pending.push(resolve)))
  vi.stubGlobal('fetch', fetcher)
  const { result, rerender, unmount } = renderHook(({ url }) => useResource<{ id: string }>(url), { initialProps: { url: '/first' } })
  rerender({ url: '/second' })
  expect(fetcher.mock.calls.length).toBe(2)
  await act(async () => { pending[1]({ ok: true, json: async () => ({ id: 'second' }) }) })
  await waitFor(() => expect(result.current.data?.id).toBe('second'))
  await act(async () => { pending[0]({ ok: true, json: async () => ({ id: 'first' }) }) })
  expect(result.current.data?.id).toBe('second')
  unmount()
})

test('API errors preserve the backend explanation', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 409, json: async () => ({ detail: 'Ambiguous experiment ID prefix' }) })))
  await expect(getJson('/experiment/abc')).rejects.toThrow('Ambiguous experiment ID prefix')
})
