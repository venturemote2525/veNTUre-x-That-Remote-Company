import { supabase } from '@/lib/supabase';
import {
  DateGroup,
  ManualLogEntry,
  ScaleLogEntry,
  ScaleLogSummary,
} from '@/types/database-types';

export async function retrieveRecentWeight(): Promise<number> {
  const { data, error } = await supabase
    .from('manual_logs')
    .select('weight')
    .not('weight', 'is', null)
    .order('logged_at', { ascending: false }) // newest first
    .limit(1)
    .maybeSingle();
  if (error) throw error;
  return data?.weight ?? null;
}

export async function retrieveRecentHeight(): Promise<number> {
  const { data, error } = await supabase
    .from('manual_logs')
    .select('height')
    .not('height', 'is', null)
    .order('logged_at', { ascending: false }) // newest first
    .limit(1)
    .maybeSingle();
  if (error) throw error;
  return data?.height ?? null;
}

export async function logDataManually(log: ManualLogEntry) {
  const { data, error } = await supabase.from('manual_logs').insert(log);
  if (error) throw error;
  return data;
}

export async function logScaleData(log: ScaleLogEntry) {
  const { data, error } = await supabase.from('scale_logs').insert(log);
  if (error) throw error;
  return data;
}

/**
 * Fetch weight/height data in groups
 */
export async function fetchWeightLogs(
  group: DateGroup,
): Promise<ScaleLogSummary[]> {
  let data;
  let error;
  switch (group) {
    case 'WEEK':
      ({ data, error } = await supabase.rpc('group_by_week'));
      break;
    case 'MONTH':
      ({ data, error } = await supabase.rpc('group_by_month'));
      break;
    case 'YEAR':
      ({ data, error } = await supabase.rpc('group_by_year'));
      break;
    default:
      throw new Error(`Unknown group: ${group}`);
  }
  if (error) throw error;
  return data;
}
