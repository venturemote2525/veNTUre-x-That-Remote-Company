import { supabase } from '@/lib/supabase';
import { UserProfileInsert } from '@/types/database-types';

export async function userSignup(email: string, password: string) {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
  });
  if (error) throw error;
  return data;
}

export async function userLogin(email: string, password: string) {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw error;
  return data;
}

export async function checkUserExists(email: string): Promise<boolean> {
  const { data, error } = await supabase.rpc('check_user_exists', {
    input_email: email.trim().toLowerCase(),
  });

  if (error) {
    console.error('Error checking user:', error);
    return false;
  }

  return data as boolean;
}

/**
 * Create new user profile from onboarding
 * @param profile User profile information
 * @returns Created data
 */
export async function createProfile(profile: UserProfileInsert) {
  const { data, error } = await supabase.from('user_profiles').insert({
    user_id: profile.user_id,
    gender: profile.gender,
    username: profile.username,
    height: profile.height,
    dob: profile.dob,
  });
  if (error) throw error;
  console.log(data);
  return data;
}
/**
 * Send password reset email
 * @param email User's email address
 * @returns void
 */

export async function resetPassword(email: string): Promise<void> {
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: 'frontend://reset-password',  
  });

  if (error) throw error;
}
