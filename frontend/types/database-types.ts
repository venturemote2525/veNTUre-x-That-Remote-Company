// -------------------- user_profiles --------------------

export interface UserProfile {
  id: string;
  user_id: string;
  username: string;
  gender: string;
  height: number;
  dob: string;
  created_at: string;
}

export type UserProfileInsert = Omit<UserProfile, 'id' | 'created_at'>;
