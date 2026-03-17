import { useState } from 'react'
import { DependencyGraphView } from '@/components/DependencyGraph'
import { ExperimentFilters } from '@/components/ExperimentFilters'

export default function GraphPage() {
  const [filters, setFilters] = useState({
    status: '',
    name_pattern: '',
    tags: '',
    limit: 0,
    started_before: '',
    started_after: '',
    ended_before: '',
    ended_after: '',
    sort_order: 'none',
  })

  const handleFilterChange = (newFilters: Partial<typeof filters>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }))
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Dependency Graph</h1>
        <p className="text-sm text-gray-500 mt-1">
          Visualize experiment dependencies. Click a node to view experiment details. Hover for parameters and metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <ExperimentFilters
            filters={filters}
            onFilterChange={handleFilterChange}
          />
        </div>

        <div className="lg:col-span-3">
          <DependencyGraphView filters={filters} />
        </div>
      </div>
    </div>
  )
}
