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
  cancelText?: string;
  onCancel?: () => void;
  confirmText?: string;
  onConfirm?: () => void;
}