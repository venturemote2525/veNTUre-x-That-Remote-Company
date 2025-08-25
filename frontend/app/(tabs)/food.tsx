import DateSelector from '@/components/DateSelector';
import MealCard from '@/components/Food/MealCard';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import dayjs from 'dayjs';
import { useState } from 'react';
import { ScrollView } from 'react-native';

const tempData = [
  { id: 1, meal: 'lunch', name: 'lunch meal', calories: 5 },
  { id: 2, meal: 'lunch', name: 'lunch meal', calories: 7 },
  { id: 3, meal: 'breakfast', name: 'breakfast meal', calories: 10 },
];

export default function FoodScreen() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const totalCalories = tempData.reduce((sum, meal) => sum + meal.calories, 0);
  return (
    <ThemedSafeAreaView edges={['top']} className="px-4">
      <DateSelector
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
      />
      <View className="button-rounded flex-row justify-between px-8">
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
            meals={tempData.filter(m => m.meal === 'breakfast')}
          />
          <MealCard
            title="Lunch"
            meals={tempData.filter(m => m.meal === 'lunch')}
          />
          <MealCard
            title="Dinner"
            meals={tempData.filter(m => m.meal === 'dinner')}
          />
          <MealCard
            title="Morning Snack"
            meals={tempData.filter(m => m.meal === 'morning-snack')}
          />
          <MealCard
            title="Afternoon Snack"
            meals={tempData.filter(m => m.meal === 'afternoon-snack')}
          />
          <MealCard
            title="Night Snack"
            meals={tempData.filter(m => m.meal === 'night-snack')}
          />
        </ScrollView>
      </View>
    </ThemedSafeAreaView>
  );
}
