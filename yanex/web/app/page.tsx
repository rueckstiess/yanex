'use client'

import { useState, useEffect } from 'react'
import { ExperimentList } from '@/components/ExperimentList'
import { ExperimentFilters } from '@/components/ExperimentFilters'
import { StatusStats } from '@/components/StatusStats'
import { Experiment } from '@/types/experiment'

export default function Home() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    status: '',
    name_pattern: '',
    tags: '',
    limit: 50,
    started_before: '',
    started_after: '',
    ended_before: '',
    ended_after: '',
    sort_order: 'newest',
  })

  const fetchExperiments = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const params = new URLSearchParams()
      if (filters.status) params.append('status', filters.status)
      if (filters.name_pattern) params.append('name_pattern', filters.name_pattern)
      if (filters.tags) params.append('tags', filters.tags)
      if (filters.limit) params.append('limit', filters.limit.toString())
      if (filters.started_before) params.append('started_before', filters.started_before)
      if (filters.started_after) params.append('started_after', filters.started_after)
      if (filters.ended_before) params.append('ended_before', filters.ended_before)
      if (filters.ended_after) params.append('ended_after', filters.ended_after)
      if (filters.sort_order) params.append('sort_order', filters.sort_order)
      
      const response = await fetch(`http://localhost:8000/api/experiments?${params}`)
      if (!response.ok) {
        throw new Error('Failed to fetch experiments')
    }
      
      const data = await response.json()
      setExperiments(data.experiments)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExperiments()
  }, [filters])

  const handleFilterChange = (newFilters: Partial<typeof filters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }))
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
        <button
          onClick={fetchExperiments}
          className="btn btn-secondary"
        >
          Refresh
        </button>
      </div>

      <StatusStats />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <ExperimentFilters
            filters={filters}
            onFilterChange={handleFilterChange}
          />
        </div>
        
        <div className="lg:col-span-3">
          <ExperimentList
            experiments={experiments}
            loading={loading}
            error={error}
          />
        </div>
      </div>
    </div>
  )
}
