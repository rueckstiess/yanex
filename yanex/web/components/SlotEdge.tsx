import { memo } from 'react'
import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
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
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 8,
  })

  const slot = (data as Record<string, any>)?.slot || ''
  const labelOffsetX = (data as Record<string, any>)?.labelOffsetX || 0
  const labelOffsetY = (data as Record<string, any>)?.labelOffsetY || 0
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
