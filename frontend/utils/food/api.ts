import { supabase } from '@/lib/supabase';
import { Meal } from '@/types/database-types';
import { decode } from 'base64-arraybuffer';
import dayjs from 'dayjs';

export async function uploadImage(
  id: string,
  userId: string,
  base64Data: string,
  foodName?: string,
) {
  const filePath = `${userId}/${Date.now()}.jpg`;

  const { error: uploadError } = await supabase.storage
    .from('meal_images')
    .upload(filePath, decode(base64Data), { contentType: 'image/jpeg' });

  if (uploadError) throw uploadError;

  // Save in meals with default values for required fields
  const { error: dbError } = await supabase.from('meals').insert({
    id,
    user_id: userId,
    image_url: filePath,
    name: foodName ?? '',
    meal: 'BREAKFAST',
    calories: 0,
    protein: 0,
    carbs: 0,
    fat: 0,
    fiber: 0,
    date: new Date().toISOString(),
  });
  if (dbError) throw dbError;
  return filePath;
}

export async function retrieveMeal(id: string): Promise<Meal> {
  const { data, error } = await supabase
    .from('meals')
    .select('*')
    .eq('id', id)
    .single();
  if (error || !data) throw error;

  const { data: signedData, error: signedError } = await supabase.storage
    .from('meal_images')
    .createSignedUrl(data.image_url, 6000);
  if (signedError) throw signedError;

  return {
    ...data,
    image_url: signedData.signedUrl,
  };
}

/**
 * Delete log in meals
 * @param id Meal ID
 */
export async function deleteMeal(id: string): Promise<void> {
  const { error } = await supabase.from('meals').delete().eq('id', id);
  if (error) throw error;
  return;
}

export async function updateMeal(mealId: string, updates: Partial<Meal>) {
  const { data, error } = await supabase
    .from('meals')
    .update(updates)
    .eq('id', mealId)
    .select()
    .single();

  if (error) {
    console.error('Error updating meal:', error);
    throw error;
  }
  return data as Meal;
}

/**
 * Retrieve meals for selected date
 * @param date Selected date
 */
export async function retrieveMeals(date: dayjs.Dayjs): Promise<Meal[]> {
  const start = date.startOf('day').toISOString();
  const end = date.endOf('day').toISOString();
  const { data, error } = await supabase
    .from('meals')
    .select('*')
    .gte('date', start)
    .lte('date', end);
  if (error) throw error;
  return data as Meal[];
}
