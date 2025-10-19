/**
 * Date formatting utilities for consistent timezone handling
 */

import { formatDistanceToNow, format } from 'date-fns'

/**
 * Parse ISO string and ensure it's treated as UTC if no timezone specified
 * This matches the Python backend behavior where timestamps are stored in UTC
 */
function parseTimestamp(isoString: string): Date {
  // If the string doesn't end with 'Z' or have a timezone offset, treat it as UTC
  if (!isoString.endsWith('Z') && !isoString.match(/[+-]\d{2}:\d{2}$/)) {
    // Add 'Z' to indicate UTC
    return new Date(isoString + 'Z')
  }
  return new Date(isoString)
}

/**
 * Format an ISO timestamp to local time for display
 * Ensures the timestamp is interpreted as UTC (matching Python backend)
 * and displayed in the user's local timezone
 */
export function formatDateTime(isoString: string): string {
  if (!isoString) return 'Unknown'
  
  try {
    const date = parseTimestamp(isoString)
    // Format as "YYYY-MM-DD HH:MM:SS" in local time
    return format(date, 'yyyy-MM-dd HH:mm:ss')
  } catch (error) {
    return isoString
  }
}

/**
 * Format timestamp as relative time (e.g., "2 hours ago")
 * Uses local time for accurate relative calculations
 */
export function formatRelativeTime(isoString: string): string {
  if (!isoString) return 'Unknown'
  
  try {
    const date = parseTimestamp(isoString)
    return formatDistanceToNow(date, { addSuffix: true })
  } catch (error) {
    return isoString
  }
}

/**
 * Format date only (no time)
 */
export function formatDate(isoString: string): string {
  if (!isoString) return 'Unknown'
  
  try {
    const date = new Date(isoString)
    return format(date, 'yyyy-MM-dd')
  } catch (error) {
    return isoString
  }
}

/**
 * Format time only (no date)
 */
export function formatTime(isoString: string): string {
  if (!isoString) return 'Unknown'
  
  try {
    const date = new Date(isoString)
    return format(date, 'HH:mm:ss')
  } catch (error) {
    return isoString
  }
}
