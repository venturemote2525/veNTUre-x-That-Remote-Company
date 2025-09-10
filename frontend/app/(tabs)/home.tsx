// home.tsx - Enhanced version
import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { useColorScheme } from 'react-native';
import { faUtensils, faChild, faPlus } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { Colors } from '@/constants/Colors';
import { AnimatedPressable, useFadeIn, GradientCard } from '@/components/AnimatedComponents';

export default function HomeScreen() {
  const router = useRouter();
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';
  const fadeIn = useFadeIn(200);

  return (
    <ThemedSafeAreaView className="flex-1 bg-background-1">
      {/* Enhanced Header */}
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
              {/* Use the same style as old code - simple icon without gradient circle */}
              <Icon as={AddIcon} size={'xl'} className="text-secondary-500" />
            </AnimatedPressable>
          }
          menuClassName="min-w-48 rounded-2xl bg-background-0 p-3 shadow-xl"
          separator={true}
        >
          <DropdownItem
            label={'Add Device'}
            onPress={() => router.push('/(user)/MyDevices')}
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
          scaleAmount={0.95}
        >
          <View className="bg-blue-50/70 rounded-2xl p-5 border border-blue-100">
            <View className="flex-row justify-between items-center">
              <View className="flex-row items-center gap-4">
                <View className="h-14 w-14 items-center justify-center rounded-full bg-blue-100">
                  <FontAwesomeIcon
                    icon={faUtensils}
                    size={24}
                    color={Colors.light.colors.primary[600]}
                  />
                </View>
                <View>
                  <Text className="font-bodyBold text-body1 text-primary-800">
                    Food
                  </Text>
                  <Text className="font-body text-body2 text-primary-600">
                    Track your meals
                  </Text>
                </View>
              </View>
              <View className="items-center">
                <Text className="font-heading text-head2 text-success-600">1,240</Text>
                <Text className="font-bodyBold text-body2 text-primary-100">
                  kcal today
                </Text>
              </View>
            </View>
          </View>
        </AnimatedPressable>
        
        {/* Body Composition Card */}
        <AnimatedPressable 
          onPress={() => router.push('/(tabs)/body')}
          scaleAmount={0.95}
        >
          <View className="bg-blue-50/70 rounded-2xl p-5 border border-blue-100">
            <View className="flex-row justify-between items-center">
              <View className="flex-row items-center gap-4">
                <View className="h-14 w-14 items-center justify-center rounded-full bg-blue-100">
                  <FontAwesomeIcon
                    icon={faChild}
                    size={24}
                    color={Colors.light.colors.primary[600]}
                  />
                </View>
                <View>
                  <Text className="font-bodyBold text-body1 text-primary-800">
                    Body Composition
                  </Text>
                  <Text className="font-body text-body2 text-primary-600">
                    Monitor your health
                  </Text>
                </View>
              </View>
            </View>
            
            {/* Stats Grid */}
            <View className="flex-row justify-between mt-4">
              <View className="items-center">
                <Text className="font-heading text-head3 text-success-600">18.5%</Text>
                <Text className="font-body text-body3 text-primary-600">Body Fat</Text>
              </View>
              <View className="items-center">
                <Text className="font-heading text-head3 text-success-600">68kg</Text>
                <Text className="font-body text-body3 text-primary-600">Weight</Text>
              </View>
              <View className="items-center">
                <Text className="font-heading text-head3 text-success-600">22.1</Text>
                <Text className="font-body text-body3 text-primary-600">BMI</Text>
              </View>
            </View>
          </View>
        </AnimatedPressable>
      </View>
    </ThemedSafeAreaView>
  );
}