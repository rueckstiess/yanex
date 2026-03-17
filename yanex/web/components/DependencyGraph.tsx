'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  MarkerType,
} from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import ExperimentNode, { type ExperimentNodeData } from './ExperimentNode'
import SlotEdge from './SlotEdge'

import '@xyflow/react/dist/style.css'

// Types for API response
interface GraphNode {
  id: string
  name: string
  status: string
  script: string
  params: Record<string, any>
  metrics: Record<string, any>
}

interface GraphEdge {
  source: string
  target: string
  slot: string
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

interface GraphFilters {
  status: string
  name_pattern: string
  tags: string
  limit: number
  started_before: string
  started_after: string
  ended_before: string
  ended_after: string
  sort_order: string
}

const NODE_WIDTH = 220
const NODE_HEIGHT = 80

const nodeTypes = { experiment: ExperimentNode }
const edgeTypes = { slot: SlotEdge }

/**
 * Auto-layout nodes using dagre (top-to-bottom).
 */
function layoutGraph(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 80, edgesep: 30 })

  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  dagre.layout(g)

  const layoutedNodes = nodes.map((node) => {
    const dagreNode = g.node(node.id)
    return {
      ...node,
      position: {
        x: dagreNode.x - NODE_WIDTH / 2,
        y: dagreNode.y - NODE_HEIGHT / 2,
      },
    }
  })

  return { nodes: layoutedNodes, edges }
}

/**
 * Convert API graph data to React Flow nodes and edges.
 */
function toReactFlow(data: GraphData): { nodes: Node[]; edges: Edge[] } {
  const rfNodes: Node[] = data.nodes.map((n) => ({
    id: n.id,
    type: 'experiment',
    position: { x: 0, y: 0 },
    data: {
      name: n.name,
      status: n.status,
      script: n.script,
      shortId: n.id.substring(0, 8),
      fullId: n.id,
      params: n.params,
      metrics: n.metrics,
    } satisfies ExperimentNodeData,
  }))

  const incomingEdgeCounts = new Map<string, number>()
  for (const edge of data.edges) {
    incomingEdgeCounts.set(
      edge.target,
      (incomingEdgeCounts.get(edge.target) || 0) + 1,
    )
  }

  const incomingEdgeIndex = new Map<string, number>()

  const rfEdges: Edge[] = data.edges.map((e, i) => {
    const count = incomingEdgeCounts.get(e.target) || 1
    const index = incomingEdgeIndex.get(e.target) || 0
    incomingEdgeIndex.set(e.target, index + 1)

    return {
      id: `e-${e.source}-${e.target}-${i}`,
      source: e.source,
      target: e.target,
      type: 'slot',
      data: {
        slot: e.slot,
        labelOffsetX: count > 1 ? (index - (count - 1) / 2) * 56 : 0,
        labelOffsetY: -18,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 16,
        height: 16,
        color: '#94a3b8',
      },
      animated: false,
    }
  })

  return layoutGraph(rfNodes, rfEdges)
}

const statusMiniMapColors: Record<string, string> = {
  completed: '#22c55e',
  failed: '#ef4444',
  running: '#eab308',
  created: '#ffffff',
  staged: '#06b6d4',
  cancelled: '#dc2626',
  unknown: '#9ca3af',
  deleted: '#d1d5db',
}

export function DependencyGraphView({ filters }: { filters: GraphFilters }) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [graphData, setGraphData] = useState<GraphData | null>(null)

  // Fetch full graph data with list-style filters
  useEffect(() => {
    const fetchGraph = async () => {
      try {
        setLoading(true)
        setError(null)

        const params = new URLSearchParams()
        if (filters.status) params.append('status', filters.status)
        if (filters.name_pattern) params.append('name_pattern', filters.name_pattern)
        if (filters.tags) params.append('tags', filters.tags)
        if (filters.limit) params.append('limit', filters.limit.toString())
        if (filters.started_before) params.append('started_before', filters.started_before)
        if (filters.started_after) params.append('started_after', filters.started_after)
        if (filters.ended_before) params.append('ended_before', filters.ended_before)
        if (filters.ended_after) params.append('ended_after', filters.ended_after)
        if (filters.sort_order) params.append('sort_order', filters.sort_order)

        const response = await fetch(`/api/graph?${params}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch graph: ${response.statusText}`)
        }
        const data: GraphData = await response.json()
        setGraphData(data)

        if (data.nodes.length === 0) {
          setNodes([])
          setEdges([])
          setLoading(false)
          return
        }

        const { nodes: layoutedNodes, edges: layoutedEdges } = toReactFlow(data)
        setNodes(layoutedNodes)
        setEdges(layoutedEdges)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }

    fetchGraph()
  }, [filters, setNodes, setEdges])

  const miniMapNodeColor = useCallback(
    (node: Node) => {
      const data = node.data as unknown as ExperimentNodeData
      return statusMiniMapColors[data.status] || '#9ca3af'
    },
    [],
  )

  const stats = useMemo(() => {
    if (!graphData) return null
    const statusCounts: Record<string, number> = {}
    graphData.nodes.forEach((n) => {
      statusCounts[n.status] = (statusCounts[n.status] || 0) + 1
    })
    return {
      total: graphData.nodes.length,
      edges: graphData.edges.length,
      completed: statusCounts.completed || 0,
      statusCounts,
    }
  }, [graphData])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[600px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">{error}</p>
        <button onClick={() => window.location.reload()} className="btn btn-primary">
          Retry
        </button>
      </div>
    )
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-400 text-5xl mb-4">⦿</div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No dependency graph</h3>
        <p className="text-gray-500 max-w-md mx-auto">
          No experiments with dependencies found. Run experiments with{' '}
          <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm">-D</code> to create
          dependency relationships.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      {stats && (
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span>{stats.total} experiments</span>
          <span className="text-gray-300">|</span>
          <span>{stats.edges} dependencies</span>
          <span className="text-gray-300">|</span>
          <span>{stats.completed} completed</span>
          <span className="text-gray-300">|</span>
          {Object.entries(stats.statusCounts)
            .filter(([status]) => status !== 'completed')
            .map(([status, count]) => (
            <span key={status} className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-full inline-block"
                style={{ backgroundColor: statusMiniMapColors[status] || '#9ca3af' }}
              />
              {count} {status}
            </span>
            ))}
        </div>
      )}

      {/* Graph */}
      <div className="card" style={{ height: '70vh' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.1}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#e5e7eb" gap={20} />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={miniMapNodeColor}
            maskColor="rgba(0, 0, 0, 0.1)"
            pannable
            zoomable
          />
        </ReactFlow>
      </div>
    </div>
  )
}
