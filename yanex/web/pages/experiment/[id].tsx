import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import { ExperimentDetails } from '@/components/ExperimentDetails'
import { Experiment } from '@/types/experiment'

export default function ExperimentPage() {
  const router = useRouter()
  const { id } = router.query
  const experimentId = id as string

  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!experimentId) return

    const fetchExperiment = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(`/api/experiments/${experimentId}`)
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

    fetchExperiment()
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
          onClick={() => router.reload()}
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
    <>
      <Head>
        <title>{experiment.name || '[Unnamed]'} - Yanex</title>
      </Head>

      <div className="space-y-6">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => router.back()}
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
    </>
  )
}
