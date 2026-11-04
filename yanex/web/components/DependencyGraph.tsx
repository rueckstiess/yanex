'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from '@xyflow/react'
import { toReactFlow, type GraphData, type GraphFilters } from '../data/graph'
import { useResource } from '../hooks/useResource'
import ExperimentNode, { type ExperimentNodeData } from './ExperimentNode'
import SlotEdge from './SlotEdge'

import '@xyflow/react/dist/style.css'

const nodeTypes = { experiment: ExperimentNode }
const edgeTypes = { slot: SlotEdge }

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
  const params = new URLSearchParams()
  Object.entries(filters).forEach(([key, value]) => {
    if (value) params.set(key, String(value))
  })
  const { data: graphData, loading, error, refresh } = useResource<GraphData>(`/api/graph?${params}`)
  const [fullscreenError, setFullscreenError] = useState<string | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const graphContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!graphData) return
    const layout = toReactFlow(graphData)
    setNodes(layout.nodes)
    setEdges(layout.edges)
  }, [graphData, setNodes, setEdges])

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === graphContainerRef.current)
    }

    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange)
  }, [])

  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement && graphContainerRef.current) {
        await graphContainerRef.current.requestFullscreen()
      } else if (document.fullscreenElement) {
        await document.exitFullscreen()
      }
    } catch {
      setFullscreenError('Fullscreen is unavailable in this browser')
    }
  }

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
        <button onClick={refresh} className="btn btn-primary">
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
      {fullscreenError && <p role="alert">{fullscreenError}</p>}
      {/* Stats bar */}
      {stats && (
        <div className="flex items-center justify-between gap-4">
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
          <button
            type="button"
            onClick={toggleFullscreen}
            className="btn btn-secondary"
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
      )}

      {/* Graph */}
      <div
        ref={graphContainerRef}
        className="card bg-white"
        style={{ height: isFullscreen ? '100vh' : '70vh' }}
      >
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
