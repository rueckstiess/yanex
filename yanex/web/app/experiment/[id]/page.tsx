'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import { ExperimentDetails } from '@/components/ExperimentDetails'
import { Experiment } from '@/types/experiment'

export default function ExperimentPage() {
  const params = useParams()
  const experimentId = params.id as string
  
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchExperiment = async () => {
      try {
        setLoading(true)
        setError(null)
        
        const response = await fetch(`http://localhost:8000/api/experiments/${experimentId}`)
        if (!response.ok) {
          throw new Error('Failed to fetch experiment')
        }
        
        const data = await response.json()
        setExperiment(data.experiment)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }

    if (experimentId) {
      fetchExperiment()
    }
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

  if (!experiment) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Experiment not found</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-4">
        <button
          onClick={() => window.history.back()}
          className="btn btn-secondary"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-bold text-gray-900">
          {experiment.name || '[Unnamed]'}
        </h1>
      </div>

      <ExperimentDetails experimentId={experimentId} />
    </div>
  )
}
