import { type Node, type Edge, MarkerType } from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import type { ExperimentNodeData } from '../components/ExperimentNode'

// Types for API response
export interface GraphNode {
  id: string
  name: string
  status: string
  script: string
  params: Record<string, any>
  metrics: Record<string, any>
}

export interface GraphEdge {
  source: string
  target: string
  slot: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphFilters {
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
const COMPONENT_GAP_Y = 140


function getGraphNodeSortKey(node: GraphNode) {
  const normalizedName = (node.name ?? '').toLowerCase()
  return `${normalizedName}\u0000${node.id}`
}

function compareGraphNodes(leftNode: GraphNode, rightNode: GraphNode) {
  return getGraphNodeSortKey(rightNode).localeCompare(getGraphNodeSortKey(leftNode))
}

function createComponentGraph(componentNodes: Node[], componentEdges: Edge[]) {
  const componentGraph = new dagre.graphlib.Graph()
  componentGraph.setDefaultEdgeLabel(() => ({}))
  componentGraph.setGraph({
    rankdir: 'LR',
    nodesep: 90,
    ranksep: 120,
    edgesep: 30,
  })

  componentNodes.forEach((node) => {
    componentGraph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })

  componentEdges.forEach((edge) => {
    componentGraph.setEdge(edge.source, edge.target)
  })

  return componentGraph
}

/**
 * Auto-layout nodes using dagre.
 *
 * - Within each connected component: left-to-right (LR)
 * - Across disconnected components: stacked top-to-bottom for clean separation
 */
export function layoutGraph(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) {
    return { nodes, edges }
  }

  const nodeOrder = new Map<string, number>()
  nodes.forEach((node, index) => nodeOrder.set(node.id, index))

  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  const adjacency = new Map<string, Set<string>>()

  nodes.forEach((node) => adjacency.set(node.id, new Set()))
  edges.forEach((edge) => {
    if (!adjacency.has(edge.source) || !adjacency.has(edge.target)) {
      return
    }
    adjacency.get(edge.source)?.add(edge.target)
    adjacency.get(edge.target)?.add(edge.source)
  })

  const visited = new Set<string>()
  const components: string[][] = []

  for (const node of nodes) {
    if (visited.has(node.id)) {
      continue
    }

    const stack = [node.id]
    const component: string[] = []
    visited.add(node.id)

    while (stack.length > 0) {
      const current = stack.pop()
      if (!current) continue
      component.push(current)

      const neighbors = adjacency.get(current)
      if (!neighbors) {
        continue
      }

      neighbors.forEach((neighbor) => {
        if (!visited.has(neighbor)) {
          visited.add(neighbor)
          stack.push(neighbor)
        }
      })
    }

    component.sort((leftId, rightId) => {
      return (nodeOrder.get(leftId) ?? 0) - (nodeOrder.get(rightId) ?? 0)
    })
    components.push(component)
  }

  components.sort((leftComponent, rightComponent) => {
    const leftFirst = nodeOrder.get(leftComponent[0]) ?? 0
    const rightFirst = nodeOrder.get(rightComponent[0]) ?? 0
    return leftFirst - rightFirst
  })

  const positionedNodesById = new Map<string, Node>()
  let currentY = 0

  for (const componentNodeIds of components) {
    const componentNodeIdSet = new Set(componentNodeIds)
    const componentNodes = componentNodeIds
      .map((nodeId) => nodeById.get(nodeId))
      .filter((node): node is Node => node !== undefined)

    const componentEdges = edges.filter(
      (edge) =>
        componentNodeIdSet.has(edge.source) && componentNodeIdSet.has(edge.target),
    )

    const incomingCounts = new Map<string, number>()
    const outgoingCounts = new Map<string, number>()
    componentNodeIds.forEach((nodeId) => {
      incomingCounts.set(nodeId, 0)
      outgoingCounts.set(nodeId, 0)
    })
    componentEdges.forEach((edge) => {
      incomingCounts.set(edge.target, (incomingCounts.get(edge.target) ?? 0) + 1)
      outgoingCounts.set(edge.source, (outgoingCounts.get(edge.source) ?? 0) + 1)
    })

    const rootNodeIds = componentNodeIds.filter(
      (nodeId) => (incomingCounts.get(nodeId) ?? 0) === 0,
    )

    let componentGraph = createComponentGraph(componentNodes, componentEdges)
    dagre.layout(componentGraph)

    const primaryRootId = [...rootNodeIds].sort((leftId, rightId) => {
      const outgoingDelta =
        (outgoingCounts.get(rightId) ?? 0) - (outgoingCounts.get(leftId) ?? 0)
      if (outgoingDelta !== 0) {
        return outgoingDelta
      }
      return (nodeOrder.get(leftId) ?? 0) - (nodeOrder.get(rightId) ?? 0)
    })[0]

    const secondaryRootFanout = [...rootNodeIds]
      .filter((nodeId) => nodeId !== primaryRootId)
      .reduce(
        (maxFanout, nodeId) => Math.max(maxFanout, outgoingCounts.get(nodeId) ?? 0),
        0,
      )

    const shouldGuideSecondaryRoots =
      rootNodeIds.length > 1 &&
      primaryRootId !== undefined &&
      (outgoingCounts.get(primaryRootId) ?? 0) >= 8 &&
      (outgoingCounts.get(primaryRootId) ?? 0) >= secondaryRootFanout * 2

    if (shouldGuideSecondaryRoots && primaryRootId) {
      const guidedGraph = createComponentGraph(componentNodes, componentEdges)
      const secondaryRoots = rootNodeIds
        .filter((nodeId) => nodeId !== primaryRootId)
        .sort((leftId, rightId) => {
          const leftY = guidedGraph.node(leftId)?.y ?? componentGraph.node(leftId)?.y ?? 0
          const rightY = guidedGraph.node(rightId)?.y ?? componentGraph.node(rightId)?.y ?? 0
          return leftY - rightY
        })

      secondaryRoots.forEach((rootId) => {
        guidedGraph.setEdge(primaryRootId, rootId, {
          minlen: 1,
          weight: 0.1,
          guide: true,
        })
      })

      dagre.layout(guidedGraph)
      componentGraph = guidedGraph
    }

    const componentTopLeftPositions = componentNodes.map((node) => {
      const dagreNode = componentGraph.node(node.id)
      return {
        id: node.id,
        x: dagreNode.x - NODE_WIDTH / 2,
        y: dagreNode.y - NODE_HEIGHT / 2,
      }
    })

    const minX = Math.min(...componentTopLeftPositions.map((p) => p.x))
    const minY = Math.min(...componentTopLeftPositions.map((p) => p.y))
    const maxY = Math.max(...componentTopLeftPositions.map((p) => p.y + NODE_HEIGHT))
    const componentHeight = maxY - minY

    componentNodes.forEach((node) => {
      const localPosition = componentTopLeftPositions.find((p) => p.id === node.id)
      if (!localPosition) {
        return
      }
      positionedNodesById.set(node.id, {
        ...node,
        position: {
          x: localPosition.x - minX,
          y: localPosition.y - minY + currentY,
        },
      })
    })

    currentY += componentHeight + COMPONENT_GAP_Y
  }

  const layoutedNodes = nodes.map((node) => positionedNodesById.get(node.id) ?? node)
  return { nodes: layoutedNodes, edges }
}

/**
 * Convert API graph data to React Flow nodes and edges.
 */
export function toReactFlow(data: GraphData): { nodes: Node[]; edges: Edge[] } {
  const sortedGraphNodes = [...data.nodes].sort(compareGraphNodes)
  const graphNodesById = new Map(sortedGraphNodes.map((node) => [node.id, node]))

  const sortedGraphEdges = [...data.edges].sort((leftEdge, rightEdge) => {
    const leftSource = graphNodesById.get(leftEdge.source)
    const rightSource = graphNodesById.get(rightEdge.source)
    if (leftSource && rightSource) {
      const sourceDelta = compareGraphNodes(leftSource, rightSource)
      if (sourceDelta !== 0) {
        return sourceDelta
      }
    }

    const leftTarget = graphNodesById.get(leftEdge.target)
    const rightTarget = graphNodesById.get(rightEdge.target)
    if (leftTarget && rightTarget) {
      const targetDelta = compareGraphNodes(leftTarget, rightTarget)
      if (targetDelta !== 0) {
        return targetDelta
      }
    }

    return leftEdge.slot.localeCompare(rightEdge.slot)
  })

  const rfNodes: Node[] = sortedGraphNodes.map((n) => ({
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
  const outgoingEdgeCounts = new Map<string, number>()
  for (const edge of sortedGraphEdges) {
    incomingEdgeCounts.set(
      edge.target,
      (incomingEdgeCounts.get(edge.target) || 0) + 1,
    )
    outgoingEdgeCounts.set(
      edge.source,
      (outgoingEdgeCounts.get(edge.source) || 0) + 1,
    )
  }

  const incomingEdgeIndex = new Map<string, number>()
  const outgoingEdgeIndex = new Map<string, number>()

  const rfEdges: Edge[] = sortedGraphEdges.map((e, i) => {
    const incomingCount = incomingEdgeCounts.get(e.target) || 1
    const incomingIndex = incomingEdgeIndex.get(e.target) || 0
    incomingEdgeIndex.set(e.target, incomingIndex + 1)

    const outgoingCount = outgoingEdgeCounts.get(e.source) || 1
    const outgoingIndex = outgoingEdgeIndex.get(e.source) || 0
    outgoingEdgeIndex.set(e.source, outgoingIndex + 1)

    return {
      id: `e-${e.source}-${e.target}-${i}`,
      source: e.source,
      target: e.target,
      type: 'slot',
      data: {
        slot: e.slot,
        labelOffsetX: -56,
        labelOffsetY:
          incomingCount > 1 ? (incomingIndex - (incomingCount - 1) / 2) * 18 : -14,
        fanoutCount: outgoingCount,
        fanoutIndex: outgoingIndex,
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

