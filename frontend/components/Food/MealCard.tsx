import { View, Text } from '@/components/Themed';
import { Pressable, Animated } from 'react-native';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { Meal } from '@/types/database-types';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import { RectButton } from 'react-native-gesture-handler';
import Ionicons from '@expo/vector-icons/Ionicons';

type MealCardProps = {
  title: string;
  meals: Meal[] | null;
  mealType: string;
  onSwipeDelete: (mealId: string) => void;
};

export default function MealCard({ title, meals, onSwipeDelete }: MealCardProps) {
  const router = useRouter();
  const totalCalories = meals?.reduce((sum, meal) => sum + meal.calories, 0);

  const renderRightActions = (progress: any, dragX: any, meal: Meal) => {
    const trans = dragX.interpolate({
      inputRange: [-80, -40, 0],
      outputRange: [0, 0.5, 1],
    });

    return (
      <View style={{ flexDirection: 'row', width: 80 }}>
        <RectButton
          style={{ 
            backgroundColor: '#ef4444',
            flex: 1,
            alignItems: 'center',
            justifyContent: 'center',
            paddingHorizontal: 16
          }}
          onPress={() => onSwipeDelete(meal.id)}
        >
          <Ionicons name="trash-outline" size={20} color="#fff" />
          <Text style={{ color: '#fff', fontSize: 12, marginTop: 4 }}>Delete</Text>
        </RectButton>
      </View>
    );
  };

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
          <Swipeable
            key={meal.id}
            renderRightActions={(progress, dragX) => renderRightActions(progress, dragX, meal)}
            friction={2}
            rightThreshold={40}
          >
            <Pressable
              onPress={() =>
                router.push({
                  pathname: '/(logging)/summary',
                  params: { mealId: meal.id, type: 'history' }
                })
              }
              className="flex-row justify-between py-3 px-4 bg-background-0"
            >
              <Text className="font-bodySemiBold text-body2 text-primary-500">
                {meal.name}
              </Text>
              <Text className="text-body2 text-primary-500">
                {meal.calories} kcal
              </Text>
            </Pressable>
          </Swipeable>
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
        className="flex-row items-center justify-between pt-2"
      >
        <Text className="font-bodyBold text-body2 text-secondary-500">
          Add Meal
        </Text>
        <Icon as={AddIcon} size={'lg'} className="text-secondary-500" />
      </Pressable>
    </View>
  );
}