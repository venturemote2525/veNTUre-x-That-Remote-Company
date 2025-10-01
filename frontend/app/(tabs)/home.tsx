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
import { fetchGroupedScaleLogs } from '@/utils/body/api';

export default function HomeScreen() {
  const router = useRouter();

  const [latestMetrics, setLatestMetrics] = useState<{
    weight: number | null;
    bmi: number | null;
    bodyFat: number | null;
  }>({
    weight: null,
    bmi: null,
    bodyFat: null,
  });
  const [loading, setLoading] = useState(true);

  //  latest
  useFocusEffect(
    useCallback(() => {
      (async () => {
        setLoading(true);
        try {
          const scaleData = await fetchGroupedScaleLogs('WEEK');
          if (scaleData && scaleData.length > 0) {
            const latestLog = scaleData
              .sort((a, b) => new Date(b.start).getTime() - new Date(a.start).getTime())[0];
            setLatestMetrics({
              weight: latestLog.average_weight || null,
              bmi: latestLog.average_bmi || null,
              bodyFat: latestLog.average_bodyfat || null,
            });
          }
        } catch (error) {
          console.error('Error fetching latest metrics:', error);
        } finally {
          setLoading(false);
        }
      })();
    }, [])
  );

  return (
    <ThemedSafeAreaView className="flex-1 bg-background-1">
      <View className="flex-row justify-between p-6 pb-4">
        <View>
          <Text className="font-heading text-head2 text-secondary-500">
            HealthSync
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
                  1,240
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
                  <Text className="text-primary-200">Monitor your health</Text>
                </View>
              </View>
            </View>

            <View className="mt-4 flex-row justify-between">
              <View className="items-center">
                <Text className="text-head3 font-heading text-success-600">
                  {loading ? '...' : latestMetrics.bodyFat !== null ? `${latestMetrics.bodyFat}%` : '-'}
                </Text>
                <Text className="text-body3 font-body text-primary-600">
                  Body Fat
                </Text>
              </View>
              <View className="items-center">
                <Text className="text-head3 font-heading text-success-600">
                  {loading ? '...' : latestMetrics.weight !== null ? `${latestMetrics.weight}kg` : '-'}
                </Text>
                <Text className="text-body3 font-body text-primary-600">
                  Weight
                </Text>
              </View>
              <View className="items-center">
                <Text className="text-head3 font-heading text-success-600">
                  {loading ? '...' : latestMetrics.bmi !== null ? `${latestMetrics.bmi}` : '-'}
                </Text>
                <Text className="text-body3 font-body text-primary-600">
                  BMI
                </Text>
              </View>
            </View>
          </View>
        </AnimatedPressable>
      </View>
    </ThemedSafeAreaView>
  );
}
