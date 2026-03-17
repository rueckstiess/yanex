'use client'

import { useState, useEffect } from 'react'
import { ExperimentDetails as ExperimentDetailsType } from '@/types/experiment'
import { formatRelativeTime } from '@/utils/dateUtils'

interface ExperimentDetailsProps {
  experimentId: string
}

function isExpandableObject(value: any): boolean {
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.keys(value).length > 0
  )
}

type ConfigRow = {
  key: string
  path: string
  value: any
  depth: number
  expandable: boolean
}

function buildConfigRows(
  obj: Record<string, any>,
  expandedPaths: Set<string>,
  prefix = '',
  depth = 0,
): ConfigRow[] {
  const rows: ConfigRow[] = []

  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key
    const expandable = isExpandableObject(value)

    rows.push({ key, path, value, depth, expandable })

    if (expandable && expandedPaths.has(path)) {
      rows.push(...buildConfigRows(value as Record<string, any>, expandedPaths, path, depth + 1))
    }
  }

  return rows
}

function formatConfigValue(value: any): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

export function ExperimentDetails({ experimentId }: ExperimentDetailsProps) {
  const [details, setDetails] = useState<{
    experiment: ExperimentDetailsType
    config: Record<string, any>
    results: Array<{
      step?: number
      timestamp: string
      [key: string]: any
    }>
    metadata: {
      environment?: any
      git?: any
      script_path?: string
    }
    artifacts: Array<{
      name: string
      size: number
      modified: number
    }>
  } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set())

  useEffect(() => {
    const fetchDetails = async () => {
      try {
        setLoading(true)
        setError(null)
        
        const response = await fetch(`/api/experiments/${experimentId}`)
        if (!response.ok) {
          throw new Error('Failed to fetch experiment details')
        }
        
        const data = await response.json()
        setDetails(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }

    fetchDetails()
  }, [experimentId])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="btn btn-primary"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!details) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Experiment not found</p>
      </div>
    )
  }

  const experiment = details.experiment
  const configRows = buildConfigRows(details.config, expandedPaths)

  const toggleExpanded = (path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card p-6">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              {experiment.name || '[Unnamed]'}
            </h2>
            <p className="text-sm text-gray-500 mt-1">ID: {experiment.id}</p>
            <div className="flex items-center space-x-4 mt-2">
              <span className={`status-badge status-${experiment.status}`}>
                {experiment.status}
              </span>
              <span className="text-sm text-gray-500">
                Created {formatRelativeTime(experiment.created_at)}
              </span>
            </div>
          </div>
        </div>

        {experiment.description && (
          <div className="mt-4">
            <p className="text-gray-700">{experiment.description}</p>
          </div>
        )}

        {experiment.tags.length > 0 && (
          <div className="mt-4">
            <div className="flex flex-wrap gap-2">
              {experiment.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Configuration */}
      {Object.keys(details.config).length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Configuration</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Parameter
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {configRows.map((row) => (
                  <tr key={row.path}>
                    <td className="px-6 py-4 text-sm font-medium text-gray-900 font-mono break-all">
                      <div
                        className="flex items-center"
                        style={{ paddingLeft: `${row.depth * 1.25}rem` }}
                      >
                        {row.expandable ? (
                          <button
                            type="button"
                            onClick={() => toggleExpanded(row.path)}
                            className="mr-2 text-gray-500 hover:text-gray-700"
                            aria-label={expandedPaths.has(row.path) ? 'Collapse' : 'Expand'}
                          >
                            {expandedPaths.has(row.path) ? '▾' : '▸'}
                          </button>
                        ) : (
                          <span className="mr-2 text-gray-300">•</span>
                        )}
                        <span>{row.key}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 break-all font-mono">
                      {row.expandable ? '{...}' : formatConfigValue(row.value)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Artifacts */}
      {details.artifacts.length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Artifacts</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Modified
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {details.artifacts.map((artifact) => (
                  <tr key={artifact.name}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {artifact.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatFileSize(artifact.size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatRelativeTime(new Date(artifact.modified * 1000).toISOString())}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <a
                        href={`/api/experiments/${experimentId}/artifacts/${artifact.name}`}
                        download
                        className="text-primary-600 hover:text-primary-900"
                      >
                        Download
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
