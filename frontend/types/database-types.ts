// -------------------- user_profiles --------------------

export interface UserProfile {
  id: string;
  user_id: string;
  username: string;
  gender: string;
  height: number;
  dob: string;
  created_at: string;
  avatar_url?: string;
}

export type UserProfileInsert = Omit<UserProfile, 'id' | 'created_at'>;

// -------------------- meals --------------------

export interface Meal {
  id: string;
  user_id: string;
  name: string;
  image_url: string;
  meal: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  date: string;
  created_at: string;
}

// -------------------- devices --------------------

export interface DBDevice {
  id: string;
  name: string;
  mac: string;
}

// -------------------- CustomAlert --------------------

export interface AlertState {
  visible: boolean;
  title: string;
  message: string;
  cancelText?: string | null;
  onCancel?: () => void;
  confirmText?: string;
  onConfirm?: () => void;
}
