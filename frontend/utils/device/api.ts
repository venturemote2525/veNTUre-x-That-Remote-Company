import { supabase } from '@/lib/supabase';
import { DBDevice } from '@/types/database-types';
import { getPhoneId } from '@/utils/helpers';

export async function retrieveDevices(): Promise<DBDevice[]> {
  const phoneId = await getPhoneId();
  const { data, error } = await supabase
    .from('devices')
    .select('id, name, mac')
    .eq('phone_id', phoneId);
  if (error) throw error;
  return data as DBDevice[];
}

export async function retrieveDeviceInfo(deviceId: string): Promise<DBDevice> {
  const { data, error } = await supabase
    .from('devices')
    .select('id, name, mac')
    .eq('id', deviceId)
    .single();
  if (error) throw error;
  return data as DBDevice;
}

/**
 * Add device to database to save mac and name
 * @param userId User's id
 * @param name Name of scale
 * @param mac MAC address of scale
 */
export async function pairDevice(userId: string, name: string, mac: string) {
  const phoneId = await getPhoneId();
  const { data, error } = await supabase.from('devices').insert({
    user_id: userId,
    name: name,
    mac: mac,
    phone_id: phoneId,
  });
  if (error) throw error;
  return data;
}

export async function unpairDevice(userId: string, mac: string) {
  const phoneId = await getPhoneId();
  const { data, error } = await supabase
    .from('devices')
    .delete()
    .eq('user_id', userId)
    .eq('mac', mac)
    .eq('phone_id', phoneId);
  if (error) throw error;
  return data;
}
