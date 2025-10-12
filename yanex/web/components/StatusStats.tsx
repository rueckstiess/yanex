'use client'

import { useState, useEffect } from 'react'
import { StatusStats as StatusStatsType } from '@/types/experiment'

export function StatusStats() {
  const [stats, setStats] = useState<StatusStatsType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status')
        if (response.ok) {
          const data = await response.json()
          setStats(data)
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [])

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="card p-4 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    )
  }

  if (!stats) {
    return null
  }

  const statusConfig = {
    completed: { label: 'Completed', color: 'text-green-600' },
    running: { label: 'Running', color: 'text-blue-600' },
    failed: { label: 'Failed', color: 'text-red-600' },
    cancelled: { label: 'Cancelled', color: 'text-yellow-600' },
    staged: { label: 'Staged', color: 'text-gray-600' },
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div className="card p-4">
        <div className="text-sm font-medium text-gray-500">Total Experiments</div>
        <div className="text-2xl font-bold text-gray-900">{stats.total_experiments}</div>
      </div>
      
      <div className="card p-4">
        <div className="text-sm font-medium text-gray-500">Archived</div>
        <div className="text-2xl font-bold text-gray-900">{stats.archived_experiments}</div>
      </div>

      {Object.entries(stats.status_counts).map(([status, count]) => {
        const config = statusConfig[status as keyof typeof statusConfig]
        if (!config) return null
        
        return (
          <div key={status} className="card p-4">
            <div className="text-sm font-medium text-gray-500">{config.label}</div>
            <div className={`text-2xl font-bold ${config.color}`}>{count}</div>
          </div>
        )
      })}
    </div>
  )
}
