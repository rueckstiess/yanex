'use client'

interface ExperimentFiltersProps {
  filters: {
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
  onFilterChange: (filters: Partial<ExperimentFiltersProps['filters']>) => void
}

export function ExperimentFilters({ filters, onFilterChange }: ExperimentFiltersProps) {
  const statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: 'staged', label: 'Staged' },
    { value: 'running', label: 'Running' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' },
  ]

  const limitOptions = [
    { value: 10, label: '10' },
    { value: 25, label: '25' },
    { value: 50, label: '50' },
    { value: 100, label: '100' },
    { value: 0, label: 'All' },
  ]

  const sortOptions = [
    { value: 'newest', label: 'Newest First' },
    { value: 'oldest', label: 'Oldest First' },
  ]

  return (
    <div className="card p-4 space-y-4">
      <h3 className="text-lg font-medium text-gray-900">Filters</h3>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Status
        </label>
        <select
          value={filters.status}
          onChange={(e) => onFilterChange({ status: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        >
          {statusOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Name Pattern
        </label>
        <input
          type="text"
          value={filters.name_pattern}
          onChange={(e) => onFilterChange({ name_pattern: e.target.value })}
          placeholder="e.g., *tuning*"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Tags
        </label>
        <input
          type="text"
          value={filters.tags}
          onChange={(e) => onFilterChange({ tags: e.target.value })}
          placeholder="e.g., production, hyperopt"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        <p className="text-xs text-gray-500 mt-1">
          Comma-separated list of tags
        </p>
      </div>

      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700">Date Range Filters</h4>
        
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Started After
            </label>
            <input
              type="datetime-local"
              value={filters.started_after}
              onChange={(e) => onFilterChange({ started_after: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Started Before
            </label>
            <input
              type="datetime-local"
              value={filters.started_before}
              onChange={(e) => onFilterChange({ started_before: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Ended After
            </label>
            <input
              type="datetime-local"
              value={filters.ended_after}
              onChange={(e) => onFilterChange({ ended_after: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Ended Before
            </label>
            <input
              type="datetime-local"
              value={filters.ended_before}
              onChange={(e) => onFilterChange({ ended_before: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>
        
        <p className="text-xs text-gray-500">
          Use date/time filters to narrow down experiments by their start and end times
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Limit
        </label>
        <select
          value={filters.limit}
          onChange={(e) => onFilterChange({ limit: parseInt(e.target.value) })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        >
          {limitOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Sort Order
        </label>
        <select
          value={filters.sort_order}
          onChange={(e) => onFilterChange({ sort_order: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        >
          {sortOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={() => onFilterChange({ 
          status: '', 
          name_pattern: '', 
          tags: '', 
          limit: 50,
          started_before: '',
          started_after: '',
          ended_before: '',
          ended_after: '',
          sort_order: 'newest'
        })}
        className="w-full btn btn-secondary"
      >
        Clear Filters
      </button>
    </div>
  )
}
