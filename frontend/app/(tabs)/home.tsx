import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { useColorScheme } from 'react-native';
import { faUtensils, faChild, faPlus } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { Colors } from '@/constants/Colors';
import {
  AnimatedPressable,
  useFadeIn,
  GradientCard,
} from '@/components/AnimatedComponents';

export default function HomeScreen() {
  const router = useRouter();
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';
  const fadeIn = useFadeIn(200);

  return (
    <ThemedSafeAreaView className="flex-1 bg-background-1">
      {/* Enhanced Header */}
      <View className="flex-row justify-between px-6 py-2 mb-2">
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
              {/* Use the same style as old code - simple icon without gradient circle */}
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
            icon={faPlus}
            itemTextClassName="text-primary-500"
          />
        </CustomDropdown>
      </View>

      {/* Main Content with Animations */}
      <View className="flex-1 gap-6 px-6" style={fadeIn}>
        {/* Food Card with Enhanced Animation */}
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

        {/* Body Composition Card */}
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

            {/* Stats Grid */}
            <View className="mt-4 flex-row justify-between">
              <View className="items-center">
                <Text className="text-head3 font-heading text-success-600">
                  18.5%
                </Text>
                <Text className="text-body3 font-body text-primary-600">
                  Body Fat
                </Text>
              </View>
              <View className="items-center">
                <Text className="text-head3 font-heading text-success-600">
                  68kg
                </Text>
                <Text className="text-body3 font-body text-primary-600">
                  Weight
                </Text>
              </View>
              <View className="items-center">
                <Text className="text-head3 font-heading text-success-600">
                  22.1
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
