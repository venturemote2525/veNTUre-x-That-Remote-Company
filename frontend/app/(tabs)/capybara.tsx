import { ThemedSafeAreaView, View, Text } from '@/components/Themed';
import { AnimatedPressable, BouncingIcon } from '@/components/AnimatedComponents';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { faOtter } from '@fortawesome/free-solid-svg-icons';

export default function CapybaraScreen() {
  return (
    <ThemedSafeAreaView className="items-center justify-center">
      <BouncingIcon>
        <FontAwesomeIcon
          icon={faOtter}
          size={64}
          color="#3b82f6"
        />
      </BouncingIcon>
      <Text className="font-bodyBold text-head2 text-primary-500 mt-4">
        Capybara Tab 🦫
      </Text>
      <Text className="text-body1 text-primary-500 mb-8">
        This is your permanent Capybara tab.
      </Text>
      
      <AnimatedPressable 
        className="bg-secondary-500 px-6 py-3 rounded-full"
        onPress={() => console.log('Capybara pressed!')}
      >
        <Text className="text-white font-bodyBold">Pet the Capybara</Text>
      </AnimatedPressable>
    </ThemedSafeAreaView>
  );
}