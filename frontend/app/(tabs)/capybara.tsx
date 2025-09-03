import { ThemedSafeAreaView, View, Text } from '@/components/Themed';

export default function CapybaraScreen() {
  return (
    <ThemedSafeAreaView className="items-center justify-center">
      <Text className="font-bodyBold text-head2 text-primary-500">
        Capybara Tab 🦫
      </Text>
      <Text className="text">This is your permanent Capybara tab.</Text>
    </ThemedSafeAreaView>
  );
}
