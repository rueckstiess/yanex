'use client'

import { useEffect, useState } from 'react'

function pad2(value: number): string {
  return String(value).padStart(2, '0')
}

function getLocalDateTimeParts(value: string): { date: string; time: string } {
  if (!value) return { date: '', time: '' }

  const dateObj = new Date(value.endsWith('Z') ? value : `${value}Z`)
  if (Number.isNaN(dateObj.getTime())) return { date: '', time: '' }

  const date = `${dateObj.getFullYear()}-${pad2(dateObj.getMonth() + 1)}-${pad2(dateObj.getDate())}`
  const time = `${pad2(dateObj.getHours())}:${pad2(dateObj.getMinutes())}`
  return { date, time }
}

function localDateTimeToUtcString(date: string, time: string): string | null {
  if (!date) return null

  const [year, month, day] = date.split('-').map(Number)
  const [hours, minutes] = (time || '00:00').split(':').map(Number)

  if (
    Number.isNaN(year) ||
    Number.isNaN(month) ||
    Number.isNaN(day) ||
    Number.isNaN(hours) ||
    Number.isNaN(minutes)
  ) {
    return null
  }

  const localDateTime = new Date(year, month - 1, day, hours, minutes, 0)
  return localDateTime.toISOString().split('.')[0]
}

function isValidDateInput(date: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return false

  const [year, month, day] = date.split('-').map(Number)
  const dateObj = new Date(year, month - 1, day)

  return (
    dateObj.getFullYear() == year &&
    dateObj.getMonth() == month - 1 &&
    dateObj.getDate() == day
  )
}

function getTodayLocalDate(): string {
  const now = new Date()
  return `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`
}

interface FilterValues {
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

interface ExperimentFiltersProps {
  filters: FilterValues
  onFilterChange: (filters: Partial<FilterValues>) => void
  showSortOrder?: boolean
  clearFilters?: FilterValues
}

const DEFAULT_CLEAR_FILTERS: FilterValues = {
  status: '',
  name_pattern: '',
  tags: '',
  limit: 50,
  started_before: '',
  started_after: '',
  ended_before: '',
  ended_after: '',
  sort_order: 'none',
}

export function ExperimentFilters({
  filters,
  onFilterChange,
  showSortOrder = true,
  clearFilters = DEFAULT_CLEAR_FILTERS,
}: ExperimentFiltersProps) {
  const [dateFilterType, setDateFilterType] = useState<'started' | 'ended'>('started')
  const [afterDateInput, setAfterDateInput] = useState('')
  const [afterTimeInput, setAfterTimeInput] = useState('')
  const [beforeDateInput, setBeforeDateInput] = useState('')
  const [beforeTimeInput, setBeforeTimeInput] = useState('')

  const activeAfter = dateFilterType === 'started' ? filters.started_after : filters.ended_after
  const activeBefore = dateFilterType === 'started' ? filters.started_before : filters.ended_before

  useEffect(() => {
    const afterParts = getLocalDateTimeParts(activeAfter)
    const beforeParts = getLocalDateTimeParts(activeBefore)
    setAfterDateInput(afterParts.date)
    setAfterTimeInput(afterParts.time)
    setBeforeDateInput(beforeParts.date)
    setBeforeTimeInput(beforeParts.time)
  }, [dateFilterType, activeAfter, activeBefore])

  const commitBoundValue = (
    bound: 'after' | 'before',
    dateValue: string,
    timeValue: string,
  ) => {
    const key =
      dateFilterType === 'started'
        ? bound === 'after'
          ? 'started_after'
          : 'started_before'
        : bound === 'after'
          ? 'ended_after'
          : 'ended_before'

    if (!dateValue) {
      onFilterChange({ [key]: '' })
      return
    }

    if (!isValidDateInput(dateValue)) {
      return
    }

    const utcString = localDateTimeToUtcString(dateValue, timeValue || '00:00')
    if (!utcString) return
    onFilterChange({ [key]: utcString })
  }
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
                type="text"
                inputMode="numeric"
                pattern="\d{4}-\d{2}-\d{2}"
                placeholder="YYYY-MM-DD"
                value={afterDateInput}
                onChange={(e) => {
                  const nextDate = e.target.value
                  setAfterDateInput(nextDate)
                  if (isValidDateInput(nextDate)) {
                    commitBoundValue('after', nextDate, afterTimeInput)
                  }
                }}
                onBlur={() => {
                  if (!afterDateInput) {
                    commitBoundValue('after', '', afterTimeInput)
                    return
                  }

                  if (isValidDateInput(afterDateInput)) {
                    commitBoundValue('after', afterDateInput, afterTimeInput)
                    return
                  }

                  setAfterDateInput(getLocalDateTimeParts(activeAfter).date)
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
              />
              <input
                type="time"
                value={afterTimeInput}
                onChange={(e) => {
                  const nextTime = e.target.value
                  setAfterTimeInput(nextTime)
                  if (afterDateInput.length === 10 && nextTime.length === 5) {
                    commitBoundValue('after', afterDateInput, nextTime)
                  }
                }}
                onBlur={() => {
                  if (afterDateInput.length === 10) {
                    commitBoundValue('after', afterDateInput, afterTimeInput)
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
                type="text"
                inputMode="numeric"
                pattern="\d{4}-\d{2}-\d{2}"
                placeholder="YYYY-MM-DD"
                value={beforeDateInput}
                onChange={(e) => {
                  const nextDate = e.target.value
                  setBeforeDateInput(nextDate)
                  if (isValidDateInput(nextDate)) {
                    commitBoundValue('before', nextDate, beforeTimeInput)
                  }
                }}
                onBlur={() => {
                  if (!beforeDateInput) {
                    commitBoundValue('before', '', beforeTimeInput)
                    return
                  }

                  if (isValidDateInput(beforeDateInput)) {
                    commitBoundValue('before', beforeDateInput, beforeTimeInput)
                    return
                  }

                  setBeforeDateInput(getLocalDateTimeParts(activeBefore).date)
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-transparent"
              />
              <input
                type="time"
                value={beforeTimeInput}
                onChange={(e) => {
                  const nextTime = e.target.value
                  setBeforeTimeInput(nextTime)
                  if (beforeDateInput.length === 10 && nextTime.length === 5) {
                    commitBoundValue('before', beforeDateInput, nextTime)
                  }
                }}
                onBlur={() => {
                  if (beforeDateInput.length === 10) {
                    commitBoundValue('before', beforeDateInput, beforeTimeInput)
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

      {showSortOrder && (
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
      )}

      <button
        onClick={() => onFilterChange(clearFilters)}
        className="w-full btn btn-secondary"
      >
        Clear Filters
      </button>
    </div>
  )
}
