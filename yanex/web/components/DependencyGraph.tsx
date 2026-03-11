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
  target: string
}

interface PickerExperiment {
  id: string
  name: string
  status: string
  script: string
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
 * Highlights the target experiment node with isTarget.
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
      isTarget: n.id === data.target,
    } satisfies ExperimentNodeData,
  }))

  const rfEdges: Edge[] = data.edges.map((e, i) => ({
    id: `e-${e.source}-${e.target}-${i}`,
    source: e.source,
    target: e.target,
    type: 'slot',
    data: { slot: e.slot },
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#94a3b8' },
    animated: false,
  }))

  return layoutGraph(rfNodes, rfEdges)
}

const statusMiniMapColors: Record<string, string> = {
  completed: '#22c55e',
  failed: '#ef4444',
  running: '#eab308',
  staged: '#3b82f6',
  cancelled: '#9ca3af',
  unknown: '#9ca3af',
  deleted: '#d1d5db',
}

// ─── Experiment Picker ───────────────────────────────────────────────

function ExperimentPicker({ onSelect }: { onSelect: (id: string) => void }) {
  const [experiments, setExperiments] = useState<PickerExperiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        setLoading(true)
        const response = await fetch('/api/graph/experiments')
        if (!response.ok) throw new Error('Failed to fetch experiments')
        const data = await response.json()
        setExperiments(data.experiments)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }
    fetchExperiments()
  }, [])

  const filtered = useMemo(() => {
    if (!search) return experiments
    const q = search.toLowerCase()
    return experiments.filter(
      (e) =>
        e.id.toLowerCase().includes(q) ||
        (e.name && e.name.toLowerCase().includes(q)) ||
        (e.script && e.script.toLowerCase().includes(q)),
    )
  }, [experiments, search])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[400px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="btn btn-primary"
        >
          Retry
        </button>
      </div>
    )
  }

  if (experiments.length === 0) {
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
    <div className="card p-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        Select an experiment to view its lineage
      </h3>

      {/* Search */}
      <input
        type="text"
        placeholder="Search by name, ID, or script..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm
          focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 mb-3"
      />

      {/* Experiment list */}
      <div className="max-h-[400px] overflow-y-auto border border-gray-200 rounded-md divide-y divide-gray-100">
        {filtered.map((exp) => (
          <button
            key={exp.id}
            onClick={() => onSelect(exp.id)}
            className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors
              flex items-center justify-between"
          >
            <div className="min-w-0">
              <div className="font-medium text-sm text-gray-900 truncate">
                {exp.name || '[Unnamed]'}
              </div>
              <div className="text-xs text-gray-500 flex items-center gap-2">
                <span className="font-mono">{exp.id.substring(0, 8)}</span>
                {exp.script && (
                  <span className="text-gray-400">&middot; {exp.script}</span>
                )}
              </div>
            </div>
            <span className={`status-badge status-${exp.status} ml-3 shrink-0`}>
              {exp.status}
            </span>
          </button>
        ))}
        {filtered.length === 0 && (
          <div className="px-4 py-6 text-center text-sm text-gray-500">
            No experiments match &quot;{search}&quot;
          </div>
        )}
      </div>

      <p className="text-xs text-gray-400 mt-2">
        {experiments.length} experiment{experiments.length !== 1 ? 's' : ''} with dependencies
      </p>
    </div>
  )
}

// ─── Main Graph View ─────────────────────────────────────────────────

export function DependencyGraphView() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Fetch lineage for selected experiment
  useEffect(() => {
    if (!selectedId) return

    const fetchGraph = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await fetch(`/api/graph?experiment_id=${selectedId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch lineage: ${response.statusText}`)
        }
        const data: GraphData = await response.json()
        setGraphData(data)

        if (data.nodes.length === 0) {
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
  }, [selectedId, setNodes, setEdges])

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
      statusCounts,
    }
  }, [graphData])

  // No experiment selected — show picker
  if (!selectedId) {
    return <ExperimentPicker onSelect={setSelectedId} />
  }

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
        <button
          onClick={() => setSelectedId(null)}
          className="btn btn-secondary mr-2"
        >
          Back
        </button>
        <button
          onClick={() => {
            setError(null)
            setSelectedId(selectedId)
          }}
          className="btn btn-primary"
        >
          Retry
        </button>
      </div>
    )
  }

  const targetNode = graphData?.nodes.find((n) => n.id === selectedId)
  const targetLabel = targetNode?.name || selectedId.substring(0, 8)

  return (
    <div className="space-y-4">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              setSelectedId(null)
              setGraphData(null)
              setNodes([])
              setEdges([])
            }}
            className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
          >
            &larr; Back
          </button>
          <h3 className="text-sm font-semibold text-gray-900">
            Lineage: {targetLabel}
          </h3>
        </div>

        {stats && (
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span>{stats.total} experiments</span>
            <span className="text-gray-300">|</span>
            <span>{stats.edges} dependencies</span>
            <span className="text-gray-300">|</span>
            {Object.entries(stats.statusCounts).map(([status, count]) => (
              <span key={status} className="flex items-center gap-1">
                <span
                  className="w-2 h-2 rounded-full inline-block"
                  style={{
                    backgroundColor: statusMiniMapColors[status] || '#9ca3af',
                  }}
                />
                {count} {status}
              </span>
            ))}
          </div>
        )}
      </div>

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
