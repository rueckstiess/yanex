import { DependencyGraphView } from '@/components/DependencyGraph'

export default function GraphPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Dependency Graph</h1>
        <p className="text-sm text-gray-500 mt-1">
          Visualize experiment dependencies. Click a node to view experiment details. Hover for parameters and metrics.
        </p>
      </div>
      <DependencyGraphView />
    </div>
  )
}
