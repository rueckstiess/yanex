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
              text-[10px] text-gray-500 font-medium pointer-events-none"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
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
