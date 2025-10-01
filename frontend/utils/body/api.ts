import { supabase } from '@/lib/supabase';
import {
  BodyLogDisplay,
  DateGroup,
  ManualLog,
  ManualLogEntry,
  ManualLogSummary,
  ScaleLog,
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
 * Fetch weight, body fat, bmi data in groups for scale logs
 */
export async function fetchGroupedScaleLogs(
  group: DateGroup,
): Promise<ScaleLogSummary[]> {
  let data;
  let error;
  switch (group) {
    case 'WEEK':
      ({ data, error } = await supabase.rpc('group_by_week_scale'));
      break;
    case 'MONTH':
      ({ data, error } = await supabase.rpc('group_by_month_scale'));
      break;
    case 'YEAR':
      ({ data, error } = await supabase.rpc('group_by_year_scale'));
      break;
    default:
      throw new Error(`Unknown group: ${group}`);
  }
  if (error) throw error;
  return data;
}

/**
 * Fetch weight, height data in groups for manual logs
 */
export async function fetchGroupedManualLogs(
  group: DateGroup,
): Promise<ManualLogSummary[]> {
  let data;
  let error;
  switch (group) {
    case 'WEEK':
      ({ data, error } = await supabase.rpc('group_by_week_manual'));
      break;
    case 'MONTH':
      ({ data, error } = await supabase.rpc('group_by_month_manual'));
      break;
    case 'YEAR':
      ({ data, error } = await supabase.rpc('group_by_year_manual'));
      break;
    default:
      throw new Error(`Unknown group: ${group}`);
  }
  if (error) throw error;
  return data;
}

export async function fetchAllManualLogs(): Promise<ManualLog[]> {
  const { data, error } = await supabase.from('manual_logs').select('*');
  if (error) throw error;
  return data;
}

export async function fetchAllScaleLogs(): Promise<ScaleLog[]> {
  const { data, error } = await supabase.from('scale_logs').select('*');
  if (error) throw error;
  return data;
}

export async function fetchRecentScaleLog(): Promise<ScaleLog> {
  const { data, error } = await supabase
    .from('scale_logs')
    .select('*')
    .order('created_at')
    .limit(1)
    .single();
  if (error) throw error;
  return data;
}

export async function fetchRecentManualLog(): Promise<ManualLog[]> {
  const { data: heightLog, error: heightError } = await supabase
    .from('manual_logs')
    .select('*')
    .neq('height', 0)
    .order('logged_at', { ascending: false })
    .limit(1);
  if (heightError) throw heightError;

  const { data: weightLog, error: weightError } = await supabase
    .from('manual_logs')
    .select('*')
    .neq('weight', 0)
    .order('logged_at', { ascending: false })
    .limit(1);
  if (weightError) throw weightError;

  return [...(heightLog ?? []), ...(weightLog ?? [])];
}

export async function fetchBodyLog(): Promise<BodyLogDisplay> {
  const [scale, manual] = await Promise.all([
    fetchRecentScaleLog(),
    fetchRecentManualLog(),
  ]);

  if (!scale && manual.length === 0) {
    return {
      weight: null,
      height: null,
      body_fat: null,
      bmi: null,
    };
  }

  const manualWeight = manual.find(
    log => log.weight != null && log.weight !== 0,
  );
  const manualHeight = manual.find(
    log => log.height != null && log.height !== 0,
  );

  let weight: number | null = scale?.weight ?? null;
  let height: number | null = manualHeight?.height ?? null;
  let bmi: number | null = scale?.bmi ?? null;
  let body_fat: number | null = scale?.body_fat ?? null;

  // Compare weight timestamps
  if (
    manualWeight &&
    (!scale || new Date(manualWeight.logged_at) > new Date(scale.created_at))
  ) {
    weight = manualWeight.weight ?? null;
  }

  return {
    weight,
    height,
    bmi,
    body_fat,
  };
}
