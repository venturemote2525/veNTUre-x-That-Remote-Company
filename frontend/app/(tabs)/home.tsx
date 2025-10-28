import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { faUtensils, faChild, faPlus } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { Colors } from '@/constants/Colors';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback, useState } from 'react';
import { fetchBodyLog } from '@/utils/body/api';

import { BodyLogDisplay, Meal } from '@/types/database-types';
import { retrieveMeals } from '@/utils/food/api';
import dayjs from 'dayjs';
import { Modal, Pressable } from 'react-native';
import HomeScene from '@/app/(home)/home-scene';

export default function HomeScreen() {
  const router = useRouter();
  const [bodyLog, setBodyLog] = useState<BodyLogDisplay>();
  const [meals, setMeals] = useState<Meal[] | null>(null);
  const [showScene, setShowScene] = useState<boolean>(false);
  const totalCalories =
    meals?.reduce((sum, meal) => sum + (meal.calories || 0), 0) ?? 0;

  useFocusEffect(
    useCallback(() => {
      (async () => {
        try {
          const bodyResult = await fetchBodyLog();
          setBodyLog(bodyResult);
          const mealResult = await retrieveMeals(dayjs(new Date()));
          setMeals(mealResult);
        } catch (error) {
          console.log('Failed to fetch body log', error);
        }
      })();
    }, []),
  );

  const handleBackgroundPress = () => setShowScene(true);

  return (
    <ThemedSafeAreaView className="flex-1 bg-background-1">
      <Pressable onPress={handleBackgroundPress} style={{ flex: 1 }}>
        <View className="flex-row justify-between p-6 pb-4">
          <View>
            <Text className="font-heading text-head2 text-secondary-500">
              STRIDE
            </Text>
            <Text className="font-body text-body2 text-primary-300">
              Welcome back! 👋
            </Text>
          </View>
          <CustomDropdown
            toggle={
              <AnimatedPressable>
                <Icon as={AddIcon} size={'xl'} className="text-secondary-500" />
              </AnimatedPressable>
            }
            minWidth={170}
            separator={true}>
            <DropdownItem
              label={'Add Device'}
              onPress={() => router.push('/(device)/MyDevices')}
              icon={faPlus}
              itemTextClassName="text-primary-500"
            />
            <DropdownItem
              label={'Manual Input'}
              onPress={() => router.push('/(body)/manual-logging')}
              icon={faPlus}
              itemTextClassName="text-primary-500"
            />
          </CustomDropdown>
        </View>

        <View className="flex-1 gap-6 px-6">
          <AnimatedPressable
            onPress={() => router.push('/(tabs)/food')}
            scaleAmount={0.95}>
            <View className="rounded-2xl bg-background-0 p-4">
              <View className="flex-row items-center justify-between">
                <View className="flex-row items-center gap-4">
                  <View className="h-14 w-14 items-center justify-center rounded-full bg-secondary-500/20">
                    <FontAwesomeIcon
                      icon={faUtensils}
                      size={24}
                      color={Colors.light.colors.secondary[500]}
                    />
                  </View>
                  <View>
                    <Text className="font-bodyBold text-body1 text-primary-500">
                      Food
                    </Text>
                    <Text className="text-primary-200">Track your meals</Text>
                  </View>
                </View>
                <View className="items-center">
                  <Text className="font-heading text-body1 text-success-600">
                    {totalCalories}
                  </Text>
                  <Text className="font-bodyBold text-primary-100">
                    kcal today
                  </Text>
                </View>
              </View>
            </View>
          </AnimatedPressable>

          <AnimatedPressable
            onPress={() => router.push('/(tabs)/body')}
            scaleAmount={0.95}>
            <View className="rounded-2xl bg-background-0 p-4">
              <View className="flex-row items-center justify-between">
                <View className="flex-row items-center gap-4">
                  <View className="h-14 w-14 items-center justify-center rounded-full bg-secondary-500/20">
                    <FontAwesomeIcon
                      icon={faChild}
                      size={24}
                      color={Colors.light.colors.secondary[500]}
                    />
                  </View>
                  <View>
                    <Text className="font-bodyBold text-body1 text-primary-500">
                      Body Composition
                    </Text>
                    <Text className="text-primary-200">
                      Monitor your health
                    </Text>
                  </View>
                </View>
              </View>

              <View className="mt-4 flex-row justify-between">
                <View className="items-center">
                  <Text className="text-head3 font-heading text-success-600">
                    {bodyLog?.body_fat}%
                  </Text>
                  <Text className="font-body text-body3 text-primary-600">
                    Body Fat
                  </Text>
                </View>
                <View className="items-center">
                  <Text className="text-head3 font-heading text-success-600">
                    {bodyLog?.weight}kg
                  </Text>
                  <Text className="font-body text-body3 text-primary-600">
                    Weight
                  </Text>
                </View>
                <View className="items-center">
                  <Text className="text-head3 font-heading text-success-600">
                    {bodyLog?.bmi}
                  </Text>
                  <Text className="font-body text-body3 text-primary-600">
                    BMI
                  </Text>
                </View>
              </View>
            </View>
          </AnimatedPressable>
        </View>
      </Pressable>
      <Modal
        visible={showScene}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setShowScene(false)}
      >
        <Pressable
          style={{
            flex: 1,
            backgroundColor: 'rgba(0,0,0,0.5)',
            justifyContent: 'center',
            alignItems: 'center',
          }}
          onPress={() => setShowScene(false)}
        >
          <HomeScene />
        </Pressable>
      </Modal>
    </ThemedSafeAreaView>
  );
}
