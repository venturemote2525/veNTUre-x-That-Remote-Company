import { useEffect, useState } from 'react';
import { View, Image, Pressable, ActivityIndicator, Alert } from 'react-native';
import { Text, ThemedSafeAreaView } from '@/components/Themed';
import Ionicons from '@expo/vector-icons/Ionicons';
import * as ImagePicker from 'expo-image-picker';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'expo-router';
import { supabase } from '@/lib/supabase';
import uuid from 'react-native-uuid';

export default function ProfileScreen() {
  const { profile, user, profileLoading, loading } = useAuth();
  const [photo, setPhoto] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const router = useRouter();

  const email = user?.email ?? undefined;
  const memberSinceIso = profile?.created_at ?? user?.created_at;
  const memberSince = memberSinceIso ? new Date(memberSinceIso).toLocaleDateString() : undefined;
  const isBusy = loading || profileLoading;

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!loading && !user) {
      router.replace('/login');
    }
  }, [user, loading]);

  // Load avatar initially from Supabase Storage
  useEffect(() => {
    const fetchMeals = async () => {
      const result = await retrieveMeals(selectedDate);
      setMeals(result);
    };
    fetchMeals();
  }, [selectedDate]);

  const totalCalories = meals?.reduce((sum, meal) => sum + meal.calories, 0);

  return (
    <ThemedSafeAreaView edges={['top']} className="px-4">
      <DateSelector selectedDate={selectedDate} onDateChange={setSelectedDate} />

      <View className="button-rounded flex-row justify-between px-8 my-2">
        <Text className="font-bodyBold text-body1 text-background-0"> Total </Text>
        <Text className="font-bodySemiBold text-body2 text-background-0"> {totalCalories} kcal </Text>
      </View>

      <View className="rounded-4xl flex-1 py-4">
        <ScrollView
          className="flex-1 rounded-2xl"
          showsVerticalScrollIndicator={false}
          contentContainerClassName="gap-4"
        >
          <MealCard title="Breakfast" meals={meals ? meals.filter(m => m.meal === 'BREAKFAST') : null} />
          <MealCard title="Lunch" meals={meals ? meals.filter(m => m.meal === 'LUNCH') : null} />
          <MealCard title="Dinner" meals={meals ? meals.filter(m => m.meal === 'DINNER') : null} />
          <MealCard title="Morning Snack" meals={meals ? meals.filter(m => m.meal === 'MORNING_SNACK') : null} />
          <MealCard title="Afternoon Snack" meals={meals ? meals.filter(m => m.meal === 'AFTERNOON_SNACK') : null} />
          <MealCard title="Night Snack" meals={meals ? meals.filter(m => m.meal === 'NIGHT_SNACK') : null} />
        </ScrollView>
      </View>
    </ThemedSafeAreaView>
  );
}
