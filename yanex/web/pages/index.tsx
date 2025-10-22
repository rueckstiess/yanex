import { useState, useEffect } from 'react'
import Head from 'next/head'
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
    sort_order: 'none',
  })
  const [sorting, setSorting] = useState({
    sort_by: 'created_at',
    sort_order: 'desc',
  })
  const [pagination, setPagination] = useState({
    page: 1,
    total_pages: 1,
    total: 0,
    limit: 50,
    has_next: false,
    has_prev: false,
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
      if (sorting.sort_by) params.append('sort_by', sorting.sort_by)
      if (sorting.sort_order) params.append('sort_order', sorting.sort_order)
      params.append('page', pagination.page.toString())

      const response = await fetch(`/api/experiments?${params}`)
      if (!response.ok) {
        throw new Error('Failed to fetch experiments')
      }

      const data = await response.json()
      setExperiments(data.experiments)
      setPagination({
        page: data.page,
        total_pages: data.total_pages,
        total: data.total,
        limit: data.limit,
        has_next: data.has_next,
        has_prev: data.has_prev,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExperiments()
  }, [filters, sorting, pagination.page])

  const handleFilterChange = (newFilters: Partial<typeof filters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }))
    // Reset to page 1 when filters change
    setPagination(prev => ({ ...prev, page: 1 }))
  }

  const handleSort = (field: string) => {
    setSorting(prev => {
      // If clicking the same field, toggle sort order
      if (prev.sort_by === field) {
        return {
          ...prev,
          sort_order: prev.sort_order === 'asc' ? 'desc' : 'asc'
        }
      }
      // If clicking a different field, set it as the new sort field with default desc order
      return {
        ...prev,
        sort_by: field,
        sort_order: 'desc'
      }
    })
    // Reset to page 1 when sorting changes
    setPagination(prev => ({ ...prev, page: 1 }))
  }

  const handlePageChange = (newPage: number) => {
    setPagination(prev => ({ ...prev, page: newPage }))
  }

  // Sort experiments on current page based on filter sort_order
  const getSortedExperiments = (experiments: Experiment[]) => {
    if (filters.sort_order === 'none') {
      return experiments // No sorting, return as-is
    } else if (filters.sort_order === 'oldest') {
      return [...experiments].sort((a, b) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      )
    } else {
      // newest (default)
      return [...experiments].sort((a, b) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )
    }
  }

  return (
    <>
      <Head>
        <title>Yanex - Experiment Tracker</title>
      </Head>

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
              experiments={getSortedExperiments(experiments)}
              loading={loading}
              error={error}
              sortBy={sorting.sort_by}
              sortOrder={sorting.sort_order}
              onSort={handleSort}
              pagination={pagination}
              onPageChange={handlePageChange}
            />
          </div>
        </div>
      </div>
    </>
  )
}
