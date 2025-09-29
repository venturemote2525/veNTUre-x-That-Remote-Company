import {
  DateGroup,
  GraphPoint,
  ScaleLogSummary,
  MetricType,
  MergedLogSummary,
  ManualLogSummary,
} from '@/types/database-types';

export function mergeGroupedLogs(
  scaleLogs: ScaleLogSummary[],
  manualLogs: ManualLogSummary[],
): MergedLogSummary[] {
  const map: Record<string, MergedLogSummary> = {};
  for (const log of scaleLogs) {
    if (!map[log.start]) {
      map[log.start] = {
        start: log.start,
        average_weight: log.average_weight,
        average_height: 0, // no height in scale logs
        average_bmi: log.average_bmi ?? 0,
        average_bodyfat: log.average_bodyfat ?? 0,
        entry_count: log.entry_count,
      };
    } else {
      // Merge if same start exists
      const existing = map[log.start];
      existing.average_weight =
        (existing.average_weight + log.average_weight) / 2;
      existing.average_bmi =
        ((existing.average_bmi ?? 0) + (log.average_bmi ?? 0)) / 2;
      existing.average_bodyfat =
        ((existing.average_bodyfat ?? 0) + (log.average_bodyfat ?? 0)) / 2;
      existing.entry_count += log.entry_count;
    }
  }

  // Process manual logs
  for (const log of manualLogs) {
    if (!map[log.start]) {
      map[log.start] = {
        start: log.start,
        average_weight: log.average_weight,
        average_height: log.average_height,
        average_bmi: 0, // no BMI in manual logs
        average_bodyfat: 0, // no body fat in manual logs
        entry_count: log.entry_count,
      };
    } else {
      // Merge if same start exists
      const existing = map[log.start];
      existing.average_weight =
        (existing.average_weight + log.average_weight) / 2;
      existing.average_height =
        (existing.average_height + log.average_height) / 2;
      existing.entry_count += log.entry_count;
    }
  }

  // Return sorted array by start date
  return Object.values(map).sort(
    (a, b) => new Date(a.start).getTime() - new Date(b.start).getTime(),
  );
}

export function transformScaleLogs(
  rows: MergedLogSummary[],
  group: DateGroup,
  metric: MetricType = 'weight', // default to weight
): GraphPoint[] {
  return (
    rows
      .map(row => {
        const date = new Date(row.start);
        let label = '';

        // Format label based on date group
        switch (group) {
          case 'WEEK':
            label = date.toLocaleDateString('en-UK', {
              month: 'short',
              day: 'numeric',
            });
            break;
          case 'MONTH':
            label = date.toLocaleDateString('en-UK', {
              month: 'short',
              year: 'numeric',
            });
            break;
          case 'YEAR':
            label = date.getFullYear().toString();
            break;
        }

        // Pick value based on metric
        let value: number;
        switch (metric) {
          case 'weight':
            value = row.average_weight ?? 0;
            break;
          case 'BMI':
            value = row.average_bmi ?? 0;
            break;
          case 'body_fat':
            value = row.average_bodyfat ?? 0;
            break;
          case 'height':
            value = row.average_height ?? 0;
            break;
        }
        // Set 0 as null to be filtered
        if (value === 0) return null;

        return { value, label };
      })
      // Filter out null (0 value) points
      .filter((point): point is GraphPoint => point !== null)
  );
}
