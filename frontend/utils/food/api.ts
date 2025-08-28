import { supabase } from '@/lib/supabase';
import { Meal } from '@/types/database-types';
import { decode } from 'base64-arraybuffer';

export async function uploadImage(id: string, userId: string, base64Data: string) {
    const filePath = `${userId}/${Date.now()}.jpg`;

    const { error: uploadError } = await supabase.storage
        .from('meal_images')
        .upload(filePath, decode(base64Data), { contentType: 'image/jpeg' });

    if (uploadError) throw uploadError;

    // Save in meals
    const {error: dbError} = await supabase
        .from('meals')
        .insert({
            id,
            user_id: userId,
            image_url: filePath
        })
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

    const { data: signedData, error: signedError } = await supabase
        .storage
        .from('meal_images')
        .createSignedUrl(data.image_url, 6000);
    if (signedError) throw signedError;

    return {
        ...data,
        image_url: signedData.signedUrl,
    };
}