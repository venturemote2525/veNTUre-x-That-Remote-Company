import {
  DateGroup,
  GraphPoint,
  ScaleLogSummary,
  MetricType,
} from '@/types/database-types';

export function transformScaleLogs(
  rows: ScaleLogSummary[],
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
        }
        // Set 0 as null to be filtered
        if (value === 0) return null;

        return { value, label };
      })
      // Filter out null (0 value) points
      .filter((point): point is GraphPoint => point !== null)
  );
}
