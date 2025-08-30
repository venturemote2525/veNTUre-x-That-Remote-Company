// api.ts
import { supabase } from '@/lib/supabase';
import { decode } from 'base64-arraybuffer';

export async function uploadAvatar(userId: string, base64Data: string) {
const filePath = `${userId}/${Date.now()}.png`; 
  // Upload image to Supabase Storage
  const { error: uploadError } = await supabase.storage
    .from('avatars')
    .upload(filePath, decode(base64Data), { contentType: 'image/jpeg' });

  if (uploadError) throw uploadError;

  // Update avatar_url in the correct table
  const { error: dbError } = await supabase
    .from('user_profiles')   
    .update({ avatar_url: filePath })
    .eq('user_id', userId);

  if (dbError) throw dbError;

  return filePath;
}
