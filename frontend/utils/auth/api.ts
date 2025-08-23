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
