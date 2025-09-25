import { DateGroup, GraphPoint, ScaleLogSummary } from '@/types/database-types';

export function transformScaleLogs(
  rows: ScaleLogSummary[],
  group: DateGroup,
): GraphPoint[] {
  return rows.map(row => {
    const date = new Date(row.start);
    let label = '';

    switch (group) {
      case 'WEEK':
        // e.g. "Jan 5"
        label = date.toLocaleDateString('en-UK', {
          month: 'short',
          day: 'numeric',
        });
        break;

      case 'MONTH':
        // e.g. "Jan 2025"
        label = date.toLocaleDateString('en-UK', {
          month: 'short',
          year: 'numeric',
        });
        break;

      case 'YEAR':
        // e.g. "2025"
        label = date.getFullYear().toString();
        break;
    }

    return {
      value: Number(row.average_weight),
      label,
    };
  });
}
