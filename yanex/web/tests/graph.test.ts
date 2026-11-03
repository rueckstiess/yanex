import { expect, test } from 'vitest'
import { toReactFlow } from '../data/graph'

test('layout preserves dependency direction, slots, and disconnected nodes', () => {
  const node = (id: string) => ({ id, name: id, status: 'completed', script: '', params: {}, metrics: {} })
  const graph = toReactFlow({ nodes: [node('a'), node('b'), node('c')], edges: [{ source: 'a', target: 'b', slot: 'data' }] })
  expect(graph.nodes.map(n => n.id).sort()).toEqual(['a', 'b', 'c'])
  expect(graph.edges[0]).toMatchObject({ source: 'a', target: 'b', data: { slot: 'data' } })
  const positions = Object.fromEntries(graph.nodes.map(n => [n.id, n.position]))
  expect(positions.a.x).toBeLessThan(positions.b.x)
  expect(Number.isFinite(positions.c.y)).toBe(true)
})
