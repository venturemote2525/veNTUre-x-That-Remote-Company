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

// -------------------- manual_logs --------------------

export interface ManualLogEntry {
  user_id: string;
  weight?: number;
  height?: number;
  logged_at: string;
}

// -------------------- scale_logs --------------------

export interface ScaleLog {
  id: number;
  user_id: string;
  weight: number;
  bmi: number;
  body_fat: number;
  created_at: string;
  // TODO: Add other scale log fields
}

export type ScaleLogEntry = Omit<ScaleLog, 'created_at'>;

export interface ScaleLogSummary {
  start: string;
  average_weight: number;
  average_bmi: number | null;
  average_bodyfat: number | null;
  entry_count: number;
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

// -------------------- MISC --------------------

export type DateGroup = 'WEEK' | 'MONTH' | 'YEAR';

export type MetricType = 'weight' | 'BMI' | 'body_fat';

export type GraphPoint = { value: number; label: string };
