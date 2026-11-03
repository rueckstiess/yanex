import { useResource } from '@/hooks/useResource'
import { useRouter } from 'next/router'
import Head from 'next/head'
import { ExperimentDetails } from '@/components/ExperimentDetails'
import { ExperimentDetailResponse } from '@/types/experiment'

export default function ExperimentPage() {
  const router = useRouter()
  const { id } = router.query
  const experimentId = id as string

  const { data: details, loading, error, refresh } = useResource<ExperimentDetailResponse>(
    experimentId ? `/api/experiments/${encodeURIComponent(experimentId)}` : null,
  )
  const experiment = details?.experiment

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
          onClick={refresh}
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

        {details && <ExperimentDetails details={details} />}
      </div>
    </>
  )
}
