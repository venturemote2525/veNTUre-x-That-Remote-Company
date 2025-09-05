import { supabase } from '@/lib/supabase';
import { ManualLogEntry } from '@/types/database-types';

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
