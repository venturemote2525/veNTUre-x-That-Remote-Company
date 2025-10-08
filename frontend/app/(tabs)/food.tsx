import DateSelector from '@/components/DateSelector';
import MealCard from '@/components/Food/MealCard';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import dayjs from 'dayjs';
import { useEffect, useState } from 'react';
import { ScrollView, Animated, Easing, Pressable, Alert } from 'react-native';
import { useCallback, useEffect, useState } from 'react';
import { ScrollView, Animated, Easing, Pressable } from 'react-native';
import { Meal } from '@/types/database-types';
import { retrieveMeals, deleteMeal } from '@/utils/food/api'; // Import deleteMeal
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import {
  faUtensils,
  faFire,
} from '@fortawesome/free-solid-svg-icons';
import { Colors } from '@/constants/Colors';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';

const mealIcons = {
  BREAKFAST: '☀️',
  LUNCH: '🌞',
  DINNER: '🌙',
  MORNING_SNACK: '🥐',
  AFTERNOON_SNACK: '🍎',
  NIGHT_SNACK: '🌰',
};

const mealColors = {
  BREAKFAST: '#FFD700',
  LUNCH: '#FF6B35',
  DINNER: '#4A6572',
  MORNING_SNACK: '#FFB74D',
  AFTERNOON_SNACK: '#FF8A65',
  NIGHT_SNACK: '#7986CB',
};

export default function FoodScreen() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [meals, setMeals] = useState<Meal[] | null>(null);
  const [progressAnim] = useState(new Animated.Value(0));
  const router = useRouter();

  // Animation values - same as profile screen
  const fadeAnim = useState(new Animated.Value(0))[0];
  const slideAnim = useState(new Animated.Value(50))[0];
  const scaleAnim = useState(new Animated.Value(0.95))[0];

  // Start animations when component mounts
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  useFocusEffect(
    useCallback(() => {
      (async () => {
        const result = await retrieveMeals(selectedDate);
        setMeals(result);

        const totalCalories =
          result?.reduce((sum, meal) => sum + meal.calories, 0) || 0;
        const progress = Math.min(totalCalories / 2000, 1);

        Animated.timing(progressAnim, {
          toValue: progress,
          duration: 800,
          easing: Easing.out(Easing.ease),
          useNativeDriver: false,
        }).start();
      })();
    }, [selectedDate]),
  );

    const handleDeleteMeal = async (mealId: string) => {
        Alert.alert(
            "Delete Meal",
            "Are you sure you want to delete this meal?",
            [
                {
                    text: "Cancel",
                    style: "cancel"
                },
                {
                    text: "Delete",
                    style: "destructive",
                    onPress: async () => {
                        try {
                            await deleteMeal(mealId);
                            // Refresh the meals after successful deletion
                            await fetchMeals();
                        } catch (error) {
                            console.error('Error deleting meal:', error);
                            Alert.alert("Error", "Failed to delete meal. Please try again.");
                        }
                    }
                }
            ]
        );
    };

  const totalCalories =
    meals?.reduce((sum, meal) => sum + meal.calories, 0) || 0;
  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1 bg-background-0">
      <Animated.ScrollView
        style={{
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }]
        }}
        className="flex-1 px-4"
        showsVerticalScrollIndicator={false}
      >
        <View className="flex-row items-center mb-4">
          <FontAwesomeIcon
            icon={faUtensils}
            size={24}
            color={Colors.light.colors.secondary[500]}
          />
          <Text className="ml-2 font-heading text-head2 text-secondary-500">
            Food Diary
          </Text>
        </View>

        <View className="flex-1">
          <DateSelector
            selectedDate={selectedDate}
            onDateChange={setSelectedDate}
          />

          <Animated.View
            style={{ transform: [{ scale: scaleAnim }] }}
            className="mb-4 rounded-2xl bg-secondary-500 p-5 shadow-lg"
          >
            <View className="mb-3 flex-row items-center justify-between">
              <View className="flex-row items-center">
                <FontAwesomeIcon
                  icon={faFire}
                  size={20}
                  color={Colors.light.background}
                />
                <Text className="ml-2 font-bodyBold text-body1 text-background-0">
                  Total Calories
                </Text>
              </View>
              <Text className="font-heading text-head2 text-background-0">
                {totalCalories}
                <Text className="font-body text-body3 text-background-0">
                  {' '}
                  kcal
                </Text>
              </Text>
            </View>

            <View className="h-2 overflow-hidden rounded-full bg-secondary-100">
              <Animated.View
                className={`h-full rounded-full ${totalCalories > 2000 ? 'bg-[#FF6B6B]' : 'bg-tertiary-500'}`}
                style={{
                  width: progressWidth,
                }}
              />
            </View>
            <View className="mt-1 flex-row justify-between">
              <Text className="font-body text-body3 text-primary-500">0</Text>
              <Text className="font-body text-body3 text-primary-500">
                2000
              </Text>
            </View>
          </Animated.View>

          <View className="flex-1 rounded-2xl">
            <View className="gap-4">
              {(
                [
                  'BREAKFAST',
                  'LUNCH',
                  'DINNER',
                  'MORNING_SNACK',
                  'AFTERNOON_SNACK',
                  'NIGHT_SNACK',
                ] as const
              ).map((mealType, index) => (
                <Animated.View
                  key={mealType}
                  className="relative"
                  style={{
                    opacity: fadeAnim,
                    transform: [
                      { translateY: slideAnim },
                      { scale: scaleAnim }
                    ]
                  }}
                >
                  <MealCard
                    title={
                      mealIcons[mealType] +
                      ' ' +
                      mealType
                        .split('_')
                        .map(
                          word =>
                            word.charAt(0) + word.slice(1).toLowerCase(),
                        )
                        .join(' ')
                    }
                    meals={
                      meals ? meals.filter(m => m.meal === mealType) : null
                    }
                    mealType={mealType}
                    onSwipeDelete={handleDeleteMeal}
                  />
                </Animated.View>
              ))}
            </View>
          </View>
        </View>
      </Animated.ScrollView>

      <Animated.View
        style={{ opacity: fadeAnim }}
        className="absolute bottom-6 right-6"
      >
        <Pressable
          onPress={() => router.push('/(tabs)/logging')}
          className="h-14 w-14 items-center justify-center rounded-full bg-secondary-500 shadow-lg active:opacity-80 active:scale-95">
          <Text className="text-2xl font-bold text-white">+</Text>
        </Pressable>
      </Animated.View>
    </ThemedSafeAreaView>
  );
}
