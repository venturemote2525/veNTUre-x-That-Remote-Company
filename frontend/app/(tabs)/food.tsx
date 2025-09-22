import DateSelector from '@/components/DateSelector';
import MealCard from '@/components/Food/MealCard';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import dayjs from 'dayjs';
import { useEffect, useState } from 'react';
import { ScrollView, Animated, Easing, Pressable } from 'react-native';
import { Meal } from '@/types/database-types';
import { retrieveMeals } from '@/utils/food/api';
import { LinearGradient } from 'expo-linear-gradient';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { faUtensils, faFire, faCalendar } from '@fortawesome/free-solid-svg-icons';
import { Colors } from '@/constants/Colors';
import { useRouter } from 'expo-router';

const mealIcons = {
  BREAKFAST: '☀️',
  LUNCH: '🌞',
  DINNER: '🌙',
  MORNING_SNACK: '🥐',
  AFTERNOON_SNACK: '🍎',
  NIGHT_SNACK: '🌰'
};

const mealColors = {
  BREAKFAST: '#FFD700',
  LUNCH: '#FF6B35',
  DINNER: '#4A6572',
  MORNING_SNACK: '#FFB74D',
  AFTERNOON_SNACK: '#FF8A65',
  NIGHT_SNACK: '#7986CB'
};

export default function FoodScreen() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [meals, setMeals] = useState<Meal[] | null>(null);
  const [progressAnim] = useState(new Animated.Value(0));
  const router = useRouter();

  useEffect(() => {
    const fetchMeals = async () => {
      const result = await retrieveMeals(selectedDate);
      setMeals(result);
      
      const totalCalories = result?.reduce((sum, meal) => sum + meal.calories, 0) || 0;
      const progress = Math.min(totalCalories / 2000, 1);
      
      Animated.timing(progressAnim, {
        toValue: progress,
        duration: 800,
        easing: Easing.out(Easing.ease),
        useNativeDriver: false,
      }).start();
    };
    fetchMeals();
  }, [selectedDate]);

  const totalCalories = meals?.reduce((sum, meal) => sum + meal.calories, 0) || 0;
  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%']
  });

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1">
      <View className="flex-1 px-4">
        <View className="flex-row items-center justify-between mb-4">
          <View className="flex-row items-center">
            <FontAwesomeIcon icon={faUtensils} size={24} color={Colors.light.colors.secondary[500]} />
            <Text className="font-heading text-head2 text-secondary-500 ml-2">
              Food Diary
            </Text>
          </View>
          <View className="flex-row items-center bg-secondary-100 rounded-full px-3 py-1">
            <FontAwesomeIcon icon={faCalendar} size={14} color={Colors.light.colors.secondary[500]} />
            <Text className="font-bodySemiBold text-body3 text-secondary-500 ml-1">
              {selectedDate.format('MMM D')}
            </Text>
          </View>
        </View>

        <DateSelector selectedDate={selectedDate} onDateChange={setSelectedDate} />

        <View className="bg-background-0 rounded-2xl p-5 mb-4 shadow-sm border border-primary-100">
          <View className="flex-row items-center justify-between mb-3">
            <View className="flex-row items-center">
              <FontAwesomeIcon icon={faFire} size={20} color={Colors.light.colors.secondary[500]} />
              <Text className="font-bodyBold text-body1 text-secondary-500 ml-2">
                Total Calories
              </Text>
            </View>
            <Text className="font-heading text-head2 text-secondary-500">
              {totalCalories}
              <Text className="font-body text-body3 text-primary-300"> kcal</Text>
            </Text>
          </View>
          
          <View className="h-2 bg-secondary-100 rounded-full overflow-hidden">
            <Animated.View 
              className="h-full rounded-full"
              style={{ 
                width: progressWidth,
                backgroundColor: totalCalories > 2000 ? '#FF6B6B' : Colors.light.colors.secondary[500]
              }}
            />
          </View>
          <View className="flex-row justify-between mt-1">
            <Text className="font-body text-body3 text-primary-300">0</Text>
            <Text className="font-body text-body3 text-primary-300">2000</Text>
          </View>
        </View>

        <View className="flex-1 rounded-2xl">
          <ScrollView
            className="flex-1"
            showsVerticalScrollIndicator={false}
            contentContainerStyle={{ paddingBottom: 20 }}
          >
            <View className="gap-4">
              {(['BREAKFAST', 'LUNCH', 'DINNER', 'MORNING_SNACK', 'AFTERNOON_SNACK', 'NIGHT_SNACK'] as const).map((mealType) => (
                <View key={mealType} className="relative">
                  <View className="flex-row items-center mb-2">
                    <Text className="text-2xl mr-2">{mealIcons[mealType]}</Text>
                    <Text className="font-bodyBold text-body1 text-secondary-500">
                      {mealType.split('_').map(word => 
                        word.charAt(0) + word.slice(1).toLowerCase()
                      ).join(' ')}
                    </Text>
                    <View className="ml-auto bg-secondary-100 rounded-full px-2 py-1">
                      <Text className="font-bodySemiBold text-body3 text-secondary-500">
                        {meals?.filter(m => m.meal === mealType).reduce((sum, meal) => sum + meal.calories, 0) || 0} kcal
                      </Text>
                    </View>
                  </View>
                  
                  <MealCard 
                    title={mealType.split('_').map(word => 
                      word.charAt(0) + word.slice(1).toLowerCase()
                    ).join(' ')} 
                    meals={meals ? meals.filter(m => m.meal === mealType) : null} 
                  />
                </View>
              ))}
            </View>
          </ScrollView>
        </View>

        <View className="absolute bottom-6 right-6">
          <Pressable 
            onPress={() => router.push('/(tabs)/logging')}
            className="w-14 h-14 rounded-full items-center justify-center shadow-lg bg-secondary-500"
          >
            <Text className="text-white text-2xl font-bold">+</Text>
          </Pressable>
        </View>
      </View>
    </ThemedSafeAreaView>
  );
}