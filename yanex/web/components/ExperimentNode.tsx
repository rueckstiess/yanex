import { memo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import Link from 'next/link'

export interface ExperimentNodeData {
  name: string
  status: string
  script: string
  shortId: string
  fullId: string
  params: Record<string, any>
  metrics: Record<string, any>
  isTarget?: boolean
}

const statusColors: Record<string, { bg: string; border: string; text: string; dot: string }> = {
  completed: { bg: 'bg-green-50', border: 'border-green-300', text: 'text-green-800', dot: 'bg-green-500' },
  failed: { bg: 'bg-red-50', border: 'border-red-300', text: 'text-red-800', dot: 'bg-red-500' },
  running: { bg: 'bg-yellow-50', border: 'border-yellow-300', text: 'text-yellow-800', dot: 'bg-yellow-500' },
  created: { bg: 'bg-white', border: 'border-gray-300', text: 'text-gray-800', dot: 'bg-gray-500' },
  staged: { bg: 'bg-cyan-50', border: 'border-cyan-300', text: 'text-cyan-800', dot: 'bg-cyan-500' },
  cancelled: { bg: 'bg-red-50', border: 'border-red-400', text: 'text-red-800', dot: 'bg-red-600' },
  unknown: { bg: 'bg-gray-50', border: 'border-gray-300', text: 'text-gray-600', dot: 'bg-gray-400' },
  deleted: { bg: 'bg-gray-50', border: 'border-gray-300', text: 'text-gray-400', dot: 'bg-gray-300' },
}

const statusIcons: Record<string, string> = {
  completed: '✓',
  failed: '✗',
  running: '⚡',
  created: '○',
  staged: '⏲',
  cancelled: '⊘',
  unknown: '?',
  deleted: '—',
}

function ExperimentNode({ data }: NodeProps) {
  const [showTooltip, setShowTooltip] = useState(false)
  const nodeRef = useRef<HTMLDivElement>(null)
  const nodeData = data as unknown as ExperimentNodeData
  const colors = statusColors[nodeData.status] || statusColors.unknown
  const icon = statusIcons[nodeData.status] || '?'

  const hasParams = Object.keys(nodeData.params).length > 0
  const hasMetrics = Object.keys(nodeData.metrics).length > 0
  const hasTooltipContent = hasParams || hasMetrics

  // Calculate tooltip position from node's bounding rect
  const getTooltipPosition = () => {
    if (!nodeRef.current) return { top: 0, left: 0 }
    const rect = nodeRef.current.getBoundingClientRect()
    return {
      top: rect.top - 8, // 8px gap above node
      left: rect.left + rect.width / 2,
    }
  }

  return (
    <div
      ref={nodeRef}
      className="relative"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <Handle type="target" position={Position.Left} className="!bg-gray-400 !w-2 !h-2" />
      <Link href={`/experiment/${nodeData.fullId}`}>
        <div
          className={`
            px-4 py-3 rounded-lg border-2 shadow-sm min-w-[200px]
            cursor-pointer transition-shadow hover:shadow-md
            ${colors.bg} ${colors.border}
            ${nodeData.isTarget ? 'ring-2 ring-offset-2 ring-indigo-500' : ''}
          `}
        >
          {/* Name */}
          <div className="font-semibold text-sm text-gray-900 truncate max-w-[220px]">
            {nodeData.name || '[Unnamed]'}
          </div>

          {/* ID + Status */}
          <div className="flex items-center justify-between mt-1">
            <span className="text-xs text-gray-500 font-mono">{nodeData.shortId}</span>
            <span className={`flex items-center text-xs font-medium ${colors.text}`}>
              <span className="mr-1">{icon}</span>
              {nodeData.status}
            </span>
          </div>

          {/* Script */}
          {nodeData.script && (
            <div className="text-xs text-gray-400 mt-1 truncate">
              {nodeData.script}
            </div>
          )}
        </div>
      </Link>
      <Handle type="source" position={Position.Right} className="!bg-gray-400 !w-2 !h-2" />

      {/* Tooltip rendered via portal so it's not clipped by React Flow's node stacking */}
      {showTooltip && hasTooltipContent && typeof document !== 'undefined' &&
        createPortal(
          <div
            className="fixed w-72 bg-white border border-gray-200 rounded-lg shadow-lg p-3
              text-xs pointer-events-none"
            style={{
              top: getTooltipPosition().top,
              left: getTooltipPosition().left,
              transform: 'translate(-50%, -100%)',
              zIndex: 10000,
            }}
          >
            {hasParams && (
              <div className="mb-2">
                <div className="font-semibold text-gray-700 mb-1">Parameters</div>
                <div className="space-y-0.5 max-h-32 overflow-y-auto">
                  {Object.entries(nodeData.params).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-500 truncate mr-2">{key}</span>
                      <span className="text-gray-900 font-mono truncate max-w-[140px]">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {hasMetrics && (
              <div>
                {hasParams && <div className="border-t border-gray-100 my-1.5" />}
                <div className="font-semibold text-gray-700 mb-1">Metrics</div>
                <div className="space-y-0.5 max-h-32 overflow-y-auto">
                  {Object.entries(nodeData.metrics).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-500 truncate mr-2">{key}</span>
                      <span className="text-gray-900 font-mono truncate max-w-[140px]">
                        {typeof value === 'number'
                          ? Number.isInteger(value) ? value : value.toFixed(4)
                          : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>,
          document.body,
        )
      }
    </div>
  )
}

export default memo(ExperimentNode)
