import { memo } from 'react'
import {
  BaseEdge,
  EdgeLabelRenderer,
  type EdgeProps,
} from '@xyflow/react'

function SlotEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps) {
  const edgeData = (data as Record<string, any>) || {}
  const fanoutCount = Number(edgeData.fanoutCount || 1)

  const isNearlyHorizontal = Math.abs(sourceY - targetY) <= 6

  let edgePath = `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`
  let labelX = (sourceX + targetX) / 2
  let labelY = (sourceY + targetY) / 2

  if (!isNearlyHorizontal) {
    const hasSiblingFanout = fanoutCount > 1
    const forkDistance = hasSiblingFanout ? 42 : 30

    const branchXMin = sourceX + 12
    const branchXMax = targetX - 12
    const branchX = Math.max(
      branchXMin,
      Math.min(branchXMax, sourceX + forkDistance),
    )

    edgePath = `M ${sourceX} ${sourceY} H ${branchX} V ${targetY} H ${targetX}`

    labelX = (branchX + targetX) / 2
    labelY = targetY
  }

  const slot = edgeData.slot || ''
  const labelOffsetX = edgeData.labelOffsetX || 0
  const labelOffsetY = edgeData.labelOffsetY || 0
  const targetLabelX = targetX + labelOffsetX
  const targetLabelY = targetY + labelOffsetY

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ stroke: '#94a3b8', strokeWidth: 1.5 }}
      />
      {slot && (
        <EdgeLabelRenderer>
          <div
            className="absolute bg-white px-1.5 py-0.5 rounded border border-gray-200
              text-[10px] text-gray-500 font-medium pointer-events-none nodrag nopan"
            style={{
              transform: `translate(-50%, -50%) translate(${targetLabelX}px, ${targetLabelY}px)`,
            }}
          >
            {slot}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

export default memo(SlotEdge)
