export interface Experiment {
  id: string
  name?: string
  status: 'staged' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  failed_at?: string
  cancelled_at?: string
  tags: string[]
  description?: string
  archived: boolean
  error_message?: string
  cancellation_reason?: string
}

export interface ExperimentDetails extends Experiment {
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
}

export interface StatusStats {
  total_experiments: number
  archived_experiments: number
  status_counts: Record<string, number>
}


