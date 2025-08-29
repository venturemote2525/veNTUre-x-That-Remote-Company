import DateSelector from '@/components/DateSelector';
import MealCard from '@/components/Food/MealCard';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import dayjs from 'dayjs';
import { useEffect, useState } from 'react';
import { ScrollView } from 'react-native';
import { Meal } from '@/types/database-types';
import { retrieveMeals } from '@/utils/food/api';

const tempData = [
  { id: 1, meal: 'lunch', name: 'lunch meal', calories: 5 },
  { id: 2, meal: 'lunch', name: 'lunch meal', calories: 7 },
  { id: 3, meal: 'breakfast', name: 'breakfast meal', calories: 10 },
];

export default function FoodScreen() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [meals, setMeals] = useState<Meal[] | null>(null);

  useEffect(() => {
    const fetchMeals = async () => {
      const result = await retrieveMeals(selectedDate)
      setMeals(result)
    };
    fetchMeals();
  }, [selectedDate]);


  const totalCalories = meals?.reduce((sum, meal) => sum + meal.calories, 0);

  return (
    <ThemedSafeAreaView edges={['top']} className="px-4">
      <DateSelector
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
      />
      <View className="button-rounded flex-row justify-between px-8 my-2">
        <Text className="font-bodyBold text-body1 text-background-0">
          Total
        </Text>
        <Text className="font-bodySemiBold text-body2 text-background-0">
          {totalCalories} kcal
        </Text>
      </View>
      <View className="rounded-4xl flex-1 py-4">
        <ScrollView
          className="flex-1 rounded-2xl"
          showsVerticalScrollIndicator={false}
          contentContainerClassName="gap-4">
          <MealCard
            title="Breakfast"
            meals={meals ? meals.filter(m => m.meal === 'BREAKFAST') : null}
          />
          <MealCard
            title="Lunch"
            meals={meals ? meals.filter(m => m.meal === 'LUNCH') : null}
          />
          <MealCard
            title="Dinner"
            meals={meals ? meals.filter(m => m.meal === 'DINNER') : null}
          />
          <MealCard
            title="Morning Snack"
            meals={meals ? meals.filter(m => m.meal === 'MORNING_SNACK') : null}
          />
          <MealCard
            title="Afternoon Snack"
            meals={meals ? meals.filter(m => m.meal === 'AFTERNOON_SNACK') : null}
          />
          <MealCard
            title="Night Snack"
            meals={meals ? meals.filter(m => m.meal === 'NIGHT_SNACK') : null}
          />
        </ScrollView>
      </View>
    </ThemedSafeAreaView>
  );
}
