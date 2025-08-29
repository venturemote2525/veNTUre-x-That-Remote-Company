import { View, Text } from '@/components/Themed';
import { Pressable } from 'react-native';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { Meal } from '@/types/database-types';

type MealCardProps = {
  title: string;
  meals: Meal[] | null;
};

export default function MealCard({ title, meals }: MealCardProps) {
  const router = useRouter();
  const totalCalories = meals?.reduce((sum, meal) => sum + meal.calories, 0);
  return (
    <View className="card-white">
      <View className="flex-row items-center justify-between">
        <Text className="font-bodyBold text-body1 text-primary-500">
          {title}
        </Text>
        <Text className="font-bodyBold text-body2 text-primary-500">
          {totalCalories} kcal
        </Text>
      </View>
      <View className="my-2 h-[1px] w-full bg-primary-500" />
      {meals && meals.length > 0 ? (
        meals.map(meal => (
          <Pressable
            onPress={() =>
              router.push({
                pathname: '/(logging)/summary',
                params: { mealId: meal.id, type: 'history' }
              })
            }
            key={meal.id}
            className="flex-row justify-between py-1">
            <Text className="font-bodySemiBold text-body2 text-primary-500">
              {meal.name}
            </Text>
            <Text className="text-body2 text-primary-500">
              {meal.calories} kcal
            </Text>
          </Pressable>
        ))
      ) : (
        <Text className="py-1 text-center text-body2 text-primary-300">
          No meals
        </Text>
      )}
      <Pressable
        onPress={() =>
          router.push({
            pathname: '/(tabs)/logging',
            params: { meal: title },
          })
        }
        className="flex-row items-center justify-between">
        <Text className="font-bodyBold text-body2 text-secondary-500">
          Add Meal
        </Text>
        <Icon as={AddIcon} size={'lg'} className="text-secondary-500" />
      </Pressable>
    </View>
  );
}
