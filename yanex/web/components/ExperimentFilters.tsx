'use client'

import { useState } from 'react'

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
  const [dateFilterType, setDateFilterType] = useState<'started' | 'ended'>('started')
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
    { value: 'none', label: 'None' },
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
        
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            Filter By
          </label>
          <select
            value={dateFilterType}
            onChange={(e) => {
              const newType = e.target.value as 'started' | 'ended'
              setDateFilterType(newType)
              // Clear the opposite filters when switching
              if (newType === 'started') {
                onFilterChange({ ended_after: '', ended_before: '' })
              } else {
                onFilterChange({ started_after: '', started_before: '' })
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="started">Started</option>
            <option value="ended">Ended</option>
          </select>
        </div>

        <div className="space-y-2">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              After
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="date"
                value={(() => {
                  const value = dateFilterType === 'started' ? filters.started_after : filters.ended_after
                  if (!value) return ''
                  // Convert UTC to local for display
                  const utcDate = new Date(value + 'Z')
                  return utcDate.toISOString().split('T')[0]
                })()}
                onChange={(e) => {
                  const currentValue = dateFilterType === 'started' ? filters.started_after : filters.ended_after
                  const currentTime = currentValue ? new Date(currentValue + 'Z').toISOString().split('T')[1].substring(0, 5) : '00:00'
                  
                  if (!e.target.value) {
                    // Clear the filter
                    if (dateFilterType === 'started') {
                      onFilterChange({ started_after: '' })
                    } else {
                      onFilterChange({ ended_after: '' })
                    }
                    return
                  }
                  
                  // Create local datetime and convert to UTC
                  const localDateTime = new Date(`${e.target.value}T${currentTime}`)
                  const utcString = localDateTime.toISOString().split('.')[0] // Remove milliseconds
                  
                  if (dateFilterType === 'started') {
                    onFilterChange({ started_after: utcString })
                  } else {
                    onFilterChange({ ended_after: utcString })
                  }
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
              />
              <input
                type="time"
                value={(() => {
                  const value = dateFilterType === 'started' ? filters.started_after : filters.ended_after
                  if (!value) return ''
                  // Convert UTC to local for display
                  const utcDate = new Date(value + 'Z')
                  return utcDate.toISOString().split('T')[1].substring(0, 5)
                })()}
                onChange={(e) => {
                  const currentValue = dateFilterType === 'started' ? filters.started_after : filters.ended_after
                  const currentDate = currentValue ? new Date(currentValue + 'Z').toISOString().split('T')[0] : new Date().toISOString().split('T')[0]
                  
                  // Create local datetime and convert to UTC
                  const localDateTime = new Date(`${currentDate}T${e.target.value}`)
                  const utcString = localDateTime.toISOString().split('.')[0] // Remove milliseconds
                  
                  if (dateFilterType === 'started') {
                    onFilterChange({ started_after: utcString })
                  } else {
                    onFilterChange({ ended_after: utcString })
                  }
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Before
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="date"
                value={(() => {
                  const value = dateFilterType === 'started' ? filters.started_before : filters.ended_before
                  if (!value) return ''
                  // Convert UTC to local for display
                  const utcDate = new Date(value + 'Z')
                  return utcDate.toISOString().split('T')[0]
                })()}
                onChange={(e) => {
                  const currentValue = dateFilterType === 'started' ? filters.started_before : filters.ended_before
                  const currentTime = currentValue ? new Date(currentValue + 'Z').toISOString().split('T')[1].substring(0, 5) : '00:00'
                  
                  if (!e.target.value) {
                    // Clear the filter
                    if (dateFilterType === 'started') {
                      onFilterChange({ started_before: '' })
                    } else {
                      onFilterChange({ ended_before: '' })
                    }
                    return
                  }
                  
                  // Create local datetime and convert to UTC
                  const localDateTime = new Date(`${e.target.value}T${currentTime}`)
                  const utcString = localDateTime.toISOString().split('.')[0] // Remove milliseconds
                  
                  if (dateFilterType === 'started') {
                    onFilterChange({ started_before: utcString })
                  } else {
                    onFilterChange({ ended_before: utcString })
                  }
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
              />
              <input
                type="time"
                value={(() => {
                  const value = dateFilterType === 'started' ? filters.started_before : filters.ended_before
                  if (!value) return ''
                  // Convert UTC to local for display
                  const utcDate = new Date(value + 'Z')
                  return utcDate.toISOString().split('T')[1].substring(0, 5)
                })()}
                onChange={(e) => {
                  const currentValue = dateFilterType === 'started' ? filters.started_before : filters.ended_before
                  const currentDate = currentValue ? new Date(currentValue + 'Z').toISOString().split('T')[0] : new Date().toISOString().split('T')[0]
                  
                  // Create local datetime and convert to UTC
                  const localDateTime = new Date(`${currentDate}T${e.target.value}`)
                  const utcString = localDateTime.toISOString().split('.')[0] // Remove milliseconds
                  
                  if (dateFilterType === 'started') {
                    onFilterChange({ started_before: utcString })
                  } else {
                    onFilterChange({ ended_before: utcString })
                  }
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>
        
        <p className="text-xs text-gray-500">
          Filter experiments by their {dateFilterType} time range. Times are in your local timezone.
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
          sort_order: 'none'
        })}
        className="w-full btn btn-secondary"
      >
        Clear Filters
      </button>
    </div>
  )
}